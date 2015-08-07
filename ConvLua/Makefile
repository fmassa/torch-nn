LIBOPTS = -shared
CFLAGS = -fPIC -std=gnu99 -fopenmp -O3
CC = gcc
LDFLAGS := -lTH

libim2col.so : im2col.c
	    $(CC) $< $(LIBOPTS) $(CFLAGS) $(LDFLAGS) -o $@

libsse.o : sseAdd.c
	    $(CC) $< $(LIBOPTS) $(CFLAGS) $(LDFLAGS) -o $@

clean :
	    rm *.so
