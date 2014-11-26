#define real float
#define FLT_MAX 3.40282347E+38F

void SpatialAdaptiveMaxPooling_updateOutput_frame(real *input_p,
                                                      real *output_p,
                                                      //real *indx_p, real *indy_p,
                                                      long nslices,
                                                      long iwidth, long iheight,
                                                      long owidth, long oheight)
{
  long k;
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++)
  {
    /* loop over output */
    long i, j;
    for(i = 0; i < oheight; i++)
    {
      int y_start = (int)floor((real)i / oheight * iheight);
      int y_end   = (int)ceil((real)(i + 1) / oheight * iheight);
      int kH = y_end-y_start;
//printf(" y_start %i y_end %i\n",y_start,y_end);      
      for(j = 0; j < owidth; j++)
      {
        
        int x_start = (int)floor((real)j / owidth * iwidth);
        int x_end   = (int)ceil((real)(j + 1) / owidth * iwidth);
        int kW = x_end-x_start;
//printf("  x_start %i x_end %i\n",x_start,x_end);        
        /* local pointers */
        real *ip = input_p   + k*iwidth*iheight + y_start*iwidth + x_start;
        real *op = output_p  + k*owidth*oheight + i*owidth + j;
        //real *indyp = indy_p + k*owidth*oheight + i*owidth + j;
        //real *indxp = indx_p + k*owidth*oheight + i*owidth + j;

        /* compute local max: */
        long maxindex = -1;
        real maxval = -FLT_MAX;
        long tcntr = 0;
        int x,y;
        for(y = 0; y < kH; y++)
        {
          for(x = 0; x < kW; x++)
          {
            real val = *(ip + y*iwidth + x);
            if (val > maxval)
            {
              maxval = val;
              maxindex = tcntr;
            }
            tcntr++;
          }
        }

        /* set output to local max */
        *op = maxval;

        /* store location of max (x,y) */
        //*indyp = (int)(maxindex / kW)+1;
        //*indxp = (maxindex % kW) +1;
      }
    }
  }
}

