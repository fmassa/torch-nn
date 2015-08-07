#include <TH/TH.h>
#include <omp.h>

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
  int nthreads = omp_get_max_threads();
  // do not take into account the non-divisible by nthreads elements
#pragma omp parallel for num_threads(nthreads)
  for(i=0; i<nthreads; ++i)
  {
    THDoubleVector_scale(data+i*size/nthreads,v,size/nthreads);
  }
  return;
}
