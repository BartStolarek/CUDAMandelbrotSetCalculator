COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-11.4/samples/common/inc -g
COBJS = bmpfile.o
EXES = mandelbrot 
all: ${EXES}

mandelbrot:   mandelbrot.cu ${COBJS}
	${COMPILER} ${CFLAGS} mandelbrot.cu ${COBJS}  -o mandelbrot -lm

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} -lm $< -c 

clean:
	rm -f *.o *~ ${EXES} my_mandelbrot_fractal.bmp 

run:
	mandelbrot 

run4k:
	mandelbrot -w=3840 -h=2160 -x=-0.55 y=0.6 -z=17400

run8k:
	mandelbrot -w=7680 -h=4320 -x=-0.55 -y=0.6 -z=34800

run16k:
	mandelbrot -w=15360 -h=8640 -x=-0.55 -y=0.6 -z=69600
