#include <THC/THC.h>

#define CUDA_MAX_THREADS 1024   // this is safe, in reality 256 is our limit
//#define FLT_MAX 3.40282347E+38F

extern "C"
{
void SpatialAdaptiveMaxPooling_updateOutput(THCudaTensor* input, THCudaTensor* output, THCudaTensor* indices, int kW, int kH);
}


/*
 * Description:
 *    this function maxpools an input 4D tensor along dimensions 2 and 3
 *    4D input, 4D output, 4D argmax x and y 
 */
__global__ void adaptivemaxpool(float *input, float *output, float *indices_x, float *indices_y,
                        int input_n, int input_h, int input_w,
			int output_h, int output_w)
                        //int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  //const int output_w = kW;//(input_w - kW) / dW + 1;
  //const int output_h = kH;//(input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  const int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  const int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  output = output + o*output_w*output_h;
  input = input + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // For all output pixels...
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {

    int y_start = (int)floor(float(yy) / output_h * input_h);
    int y_end   = (int)ceil(float(yy+1) / output_h * input_h);
    int kH = y_end-y_start;

    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      int x_start = (int)floor(float(xx) / output_w * input_w);
      int x_end   = (int)ceil(float(xx + 1) / output_w * input_w);

      int kW = x_end-x_start;

      // Compute the mean of the input image...
      //float *ptr_input = input + yy*dH*input_w + xx*dW;
      float *ptr_input = input + y_start*input_w + x_start;

      float *ptr_output = output + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      int argmax_x = -1;
      int argmax_y = -1;
      float max = -FLT_MAX;
      int kx, ky;
      for(ky = 0; ky < kH; ky++) {
        for(kx = 0; kx < kW; kx++) {
          float val = ptr_input[kx];
          if (val > max) {
            max = val;
            argmax_x = kx;
            argmax_y = ky;
          } 
        }
        ptr_input += input_w; // next input line
      }
      // Update output and argmax
      *ptr_output = max;
      *ptr_ind_x = argmax_x + 1;
      *ptr_ind_y = argmax_y + 1;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 */
__global__ void adaptivemaxgradinput(float *gradInput, float *gradOutput, float *indices_x, float *indices_y,
                             int input_n, int input_h, int input_w,
                             int kH, int kW, int dH, int dW)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      float z = *ptr_gradOutput;

      int argmax_x = (*ptr_ind_x)-1;
      int argmax_y = (*ptr_ind_y)-1;

      ptr_gradInput[argmax_x + argmax_y*input_w] += z;
    }
  }
}

/*
 * Description:
 *    this function computes the gradInput from weight and gradOutput
 *    when kH != dH or kW != dW (uses atomic add)
 */
__global__ void atomicadaptivemaxgradinput(
  float *gradInput, float *gradOutput, float *indices_x, float *indices_y,
  int input_n, int input_h, int input_w, int kH, int kW, int dH, int dW
)
{
  // iterators
  int xx, yy;

  // output size
  int output_w = (input_w - kW) / dW + 1;
  int output_h = (input_h - kH) / dH + 1;

  // compute offsets based on thread/block ID
  int o = blockIdx.x;
  int i = o;
  //int k = blockIdx.x % input_n;

  int xx_start = threadIdx.x;
  int xx_end = output_w;
  int xx_step = blockDim.x;

  int yy_start = blockDim.y*blockIdx.y + threadIdx.y;
  int yy_end = output_h;
  int yy_step = blockDim.y*gridDim.y;

  // select input/output plane
  gradOutput = gradOutput + o*output_w*output_h;
  gradInput = gradInput + i*input_w*input_h;
  indices_x = indices_x + o*output_w*output_h;
  indices_y = indices_y + o*output_w*output_h;

  // compute gradInput
  for(yy = yy_start; yy < yy_end; yy+=yy_step) {
    for(xx = xx_start; xx < xx_end; xx+=xx_step) {
      float *ptr_gradInput = gradInput + yy*dH*input_w + xx*dW;
      float *ptr_gradOutput = gradOutput + yy*output_w + xx;
      float *ptr_ind_x = indices_x + yy*output_w + xx;
      float *ptr_ind_y = indices_y + yy*output_w + xx;
      float z = *ptr_gradOutput;

      int argmax_x = (*ptr_ind_x)-1;
      int argmax_y = (*ptr_ind_y)-1;

      // atomic add since different threads could update same variable
      atomicAdd(&(ptr_gradInput[argmax_x + argmax_y*input_w]), z);
    }
  }
}

//static int cunn_SpatialAdaptiveMaxPooling_updateOutput(lua_State *L)
//{
//  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  int kW = luaT_getfieldcheckint(L, 1, "kW");
//  int kH = luaT_getfieldcheckint(L, 1, "kH");
  //int dW = luaT_getfieldcheckint(L, 1, "dW");
  //int dH = luaT_getfieldcheckint(L, 1, "dH");
void SpatialAdaptiveMaxPooling_updateOutput(THCudaTensor* input, THCudaTensor* output, THCudaTensor* indices, int kW, int kH)
{
//  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
//  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  float *indices_data;
  float *output_data;
  float *input_data;

  //luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D (batch) tensor expected");

  if (input->nDimension == 3) {
    long nInputCols = input->size[2];
    long nInputRows = input->size[1];
    long nInputPlane = input->size[0];
    long nOutputCols = kW;
    long nOutputRows = kH;

    //luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(input);
    input_data = THCudaTensor_data(input);

    THCudaTensor_resize3d(output, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize4d(indices, 2, nInputPlane, nOutputRows, nOutputCols);
    
    indices_data = THCudaTensor_data(indices);
    output_data = THCudaTensor_data(output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads>>> (input_data, output_data, 
                                   indices_data+nInputPlane*nOutputCols*nOutputRows, indices_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
  } else {
    long nInputCols = input->size[3];
    long nInputRows = input->size[2];
    long nInputPlane = input->size[1];
    long nbatch = input->size[0];
    long nOutputCols = kW;
    long nOutputRows = kH;

    //luaL_argcheck(L, nInputCols >= kW && nInputRows >= kH, 2, "input image smaller than kernel size");

    input = THCudaTensor_newContiguous(input);
    input_data = THCudaTensor_data(input);

    THCudaTensor_resize4d(output, nbatch, nInputPlane, nOutputRows, nOutputCols);
    THCudaTensor_resize5d(indices, 2, nbatch, nInputPlane, nOutputRows, nOutputCols);

    indices_data = THCudaTensor_data(indices);
    output_data = THCudaTensor_data(output);

    // cuda blocks & threads:
    int yblocks = (int)(16L / nInputPlane);
    yblocks = yblocks < 1 ? 1 : yblocks;
    dim3 blocks(nInputPlane*nbatch,yblocks);
    dim3 threads(32,8);

    // run maxpool kernel
    adaptivemaxpool <<<blocks, threads>>> (input_data, output_data,
                                   indices_data+nbatch*nInputPlane*nOutputCols*nOutputRows, indices_data,
                                   nInputPlane, nInputRows, nInputCols, nOutputRows, nOutputCols);
  }

  // clean
  THCudaTensor_free(input);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in SpatialMaxsampling.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  //return 1;
}
