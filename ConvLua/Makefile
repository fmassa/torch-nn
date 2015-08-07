LIBOPTS = -shared
CFLAGS = -fPIC -std=gnu99 -fopenmp -O3
CC = gcc

libim2col.so : im2col.c
	    $(CC) $< $(LIBOPTS) $(CFLAGS) -o $@

clean :
	    rm *.so
