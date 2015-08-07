nnf = {}

ffi = require("ffi")
-- Load myLib
myLib = ffi.load(paths.cwd() .. '/libim2col.so')
-- Function prototypes definition
ffi.cdef [[
void im2col(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    double* data_col);

void im2col_par(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    double* data_col);


void unfolded_copy(double *finput_data, double *input_data,
                               int kW, int kH,
                               int dW, int dH,
                               int padW, int padH,
                               int nInputPlane,
                               int inputWidth, int inputHeight,
                               int outputWidth, int outputHeight);

]]

local SpatialConvolution, parent = torch.class('nnf.SpatialConvolution','nn.SpatialConvolutionMM')

function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
  parent.__init(self,nInputPlane,nOutputPlane,kW,kH,dW,dH,padW,padH)
end

function SpatialConvolution:updateOutput(input)
  assert(input:nDimension() == 4)
  local inputWidth = input:size(4)
  local inputHeight = input:size(3)
  local outputWidth = torch.round((inputWidth + 2*self.padW - self.kW) / self.dW + 1)
  local outputHeight = torch.round((inputHeight + 2*self.padH - self.kH) / self.dH + 1)
  local batchSize = input:size(1)
  self.output:resize(batchSize, self.nOutputPlane, outputHeight * outputWidth)
  local ones = self.fgradInput
  local columns = self.finput
  if ones:nDimension() ~= 2 or ones:size(1)*ones:size(2) < outputHeight*outputWidth then
    ones:resize(outputHeight * outputWidth,1):fill(1)
  end
  --if columns:dim() == 0 then
    columns:resize(self.nInputPlane*self.kW*self.kH, outputHeight*outputWidth)
  --end
  self.output:copy(self.bias:view(1,-1,1):expandAs(self.output))
  for i=1,batchSize do
    --self.output[i]:copy(self.bias:view(-1,1):expandAs(self.output[i]))
    --self.output[i]:t():mm(ones,self.bias:view(1,self.nOutputPlane))
    --columns.nn.im2col(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    myLib.im2col(input[i]:data(),self.nInputPlane,inputHeight,inputWidth,self.kH,self.kW,self.padH,self.padW,self.dH,self.dW,columns:data())
    --(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    self.output[i]:addmm(self.weight, columns)
  end
  self.output = self.output:view(batchSize, self.nOutputPlane, outputHeight, outputWidth)
  return self.output
end

function SpatialConvolution:updateOutput2(input)
  assert(input:nDimension() == 4)
  local inputWidth = input:size(4)
  local inputHeight = input:size(3)
  local outputWidth = torch.round((inputWidth + 2*self.padW - self.kW) / self.dW + 1)
  local outputHeight = torch.round((inputHeight + 2*self.padH - self.kH) / self.dH + 1)
  local batchSize = input:size(1)
  self.output:resize(batchSize, self.nOutputPlane, outputHeight * outputWidth)
  local ones = self.fgradInput
  local columns = self.finput
  if ones:nDimension() ~= 2 or ones:size(1)*ones:size(2) < outputHeight*outputWidth then
    ones:resize(outputHeight * outputWidth,1):fill(1)
  end
  --if columns:dim() == 0 then
    columns:resize(self.nInputPlane*self.kW*self.kH, outputHeight*outputWidth)
  --end
  self.output:copy(self.bias:view(1,-1,1):expandAs(self.output))
  for i=1,batchSize do
    --self.output[i]:copy(self.bias:view(-1,1):expandAs(self.output[i]))
    --self.output[i]:t():mm(ones,self.bias:view(1,self.nOutputPlane))
    --columns.nn.im2col(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    --myLib.im2col(input[i]:data(),self.nInputPlane,inputHeight,inputWidth,self.kH,self.kW,self.padH,self.padW,self.dH,self.dW,columns:data())
    myLib.unfolded_copy(columns:data(), input[i]:data(),
                               self.kW, self.kH,
                               self.dW, self.dH,
                               self.padW, self.padH,
                               self.nInputPlane,
                               inputWidth, inputHeight,
                               outputWidth, outputHeight)

    --(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    self.output[i]:addmm(self.weight, columns)
  end
  self.output = self.output:view(batchSize, self.nOutputPlane, outputHeight, outputWidth)
  return self.output
end

function SpatialConvolution:updateOutput3(input)
  assert(input:nDimension() == 4)
  local inputWidth = input:size(4)
  local inputHeight = input:size(3)
  local outputWidth = torch.round((inputWidth + 2*self.padW - self.kW) / self.dW + 1)
  local outputHeight = torch.round((inputHeight + 2*self.padH - self.kH) / self.dH + 1)
  local batchSize = input:size(1)
  self.output:resize(batchSize, self.nOutputPlane, outputHeight * outputWidth)
  local ones = self.fgradInput
  local columns = self.finput
  if ones:nDimension() ~= 2 or ones:size(1)*ones:size(2) < outputHeight*outputWidth then
    ones:resize(outputHeight * outputWidth,1):fill(1)
  end
  --if columns:dim() == 0 then
    columns:resize(self.nInputPlane*self.kW*self.kH, outputHeight*outputWidth)
  --end
  self.output:copy(self.bias:view(1,-1,1):expandAs(self.output))
  for i=1,batchSize do
    --self.output[i]:copy(self.bias:view(-1,1):expandAs(self.output[i]))
    --self.output[i]:t():mm(ones,self.bias:view(1,self.nOutputPlane))
    --columns.nn.im2col(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    myLib.im2col_par(input[i]:data(),self.nInputPlane,inputHeight,inputWidth,self.kH,self.kW,self.padH,self.padW,self.dH,self.dW,columns:data())
    --(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    self.output[i]:addmm(self.weight, columns)
  end
  self.output = self.output:view(batchSize, self.nOutputPlane, outputHeight, outputWidth)
  return self.output
end

function SpatialConvolution:updateOutput4(input)
  assert(input:nDimension() == 4)
  local inputWidth = input:size(4)
  local inputHeight = input:size(3)
  local outputWidth = torch.round((inputWidth + 2*self.padW - self.kW) / self.dW + 1)
  local outputHeight = torch.round((inputHeight + 2*self.padH - self.kH) / self.dH + 1)
  local batchSize = input:size(1)
  self.output:resize(batchSize, self.nOutputPlane, outputHeight * outputWidth)
  local ones = self.fgradInput
  local columns = self.finput
  if ones:nDimension() ~= 2 or ones:size(1)*ones:size(2) < outputHeight*outputWidth then
    ones:resize(outputHeight * outputWidth,1):fill(1)
  end
  --if columns:dim() == 0 then
    columns:resize(self.nInputPlane*self.kW*self.kH, outputHeight*outputWidth)
  --end
  --self.output:copy(self.bias:view(1,-1,1):expandAs(self.output))
  for i=1,batchSize do
    for j=1,self.nOutputPlane do
     self.output[i][j]:fill(self.bias[j]) 
    end
    --self.output[i]:copy(self.bias:view(-1,1):expandAs(self.output[i]))
    --self.output[i]:t():mm(ones,self.bias:view(1,self.nOutputPlane))
    --columns.nn.im2col(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    --myLib.im2col(input[i]:data(),self.nInputPlane,inputHeight,inputWidth,self.kH,self.kW,self.padH,self.padW,self.dH,self.dW,columns:data())
    myLib.unfolded_copy(columns:data(), input[i]:data(),
                               self.kW, self.kH,
                               self.dW, self.dH,
                               self.padW, self.padH,
                               self.nInputPlane,
                               inputWidth, inputHeight,
                               outputWidth, outputHeight)

    --(columns, input[i], self.kW, self.kH, self.dW, self.dH, self.padW, self.padH)
    self.output[i]:addmm(self.weight, columns)
  end
  self.output = self.output:view(batchSize, self.nOutputPlane, outputHeight, outputWidth)
  return self.output
end

