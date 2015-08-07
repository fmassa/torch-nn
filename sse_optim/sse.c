#include <TH/TH.h>

void sseMul(THDoubleTensor* t1, double v)
{
  double* data = THDoubleTensor_data(t1);
  long size = THDoubleTensor_nElement(t1);
  THDoubleVector_scale(data,v,size);
  return;
}


void Mul(THDoubleTensor* t1, double v)
{
  double* data = THDoubleTensor_data(t1);
  long size = THDoubleTensor_nElement(t1);
  long i;
  for(i=0; i<size; ++i)
  {
    data[i]*=v;
  }
  return;
}

void parMul(THDoubleTensor* t1, double v)
{
  double* data = THDoubleTensor_data(t1);
  long size = THDoubleTensor_nElement(t1);
  long i;
#pragma omp parallel for private(i) //if(sz > TH_OMP_OVERHEAD_THRESHOLD) private(i)
  for(i=0; i<size; ++i)
  {
    data[i]*=v;
  }
  return;
}

void parsseMul(THDoubleTensor* t1, double v)
{
  double* data = THDoubleTensor_data(t1);
  long size = THDoubleTensor_nElement(t1);
  long i;
#pragma omp parallel for num_threads(4)
  for(i=0; i<4; ++i)
  {
    THDoubleVector_scale(data+i*size/4,v,size/4);
    //data[i]*=v;
  }
  return;
}
