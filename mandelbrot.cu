/**********************************
 * @file mandelbrot.cu
 * @author Bart Stolarek (bstolare@myune.edu.au)
 * @brief This program will...
 * @version 0.1
 * @date 2022-09-04
 * 
 * @copyright Copyright (c) 2022
 * 
 */

// Standard libraries
#include <stdlib.h>
#include <stdio.h>
#include "bmpfile.h"

//CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

// Pre-defined defaults incase user don't input commands
#define WIDTH 1920
#define HEIGHT 1080
#define MAX_ITER 1000
#define ZOOM 8700.0
#define XCENTER -0.55
#define YCENTER 0.6


/*Colour Values*/
#define COLOUR_DEPTH 255
#define COLOUR_MAX 240.0
#define GRADIENT_COLOUR_MAX 230.0

// File name
#define FILENAME "my_mandelbrot_fractal.bmp"


// Matrix structure for mandelbrot grid
typedef struct
{
    int width;
    int height;
    int total_pixels;
    double xcenter;
    double ycenter;
    double zoom;
    int *elements;
} Matrix;

/**
 * Computes the color gradiant
 * color: the output vector
 * x: the gradiant (beetween 0 and 360)
 * min and max: variation of the RGB channels (Move3D 0 -> 1)
 * Check wiki for more details on the colour science: en.wikipedia.org/wiki/HSL_and_HSV
 */
void GroundColorMix(double *color, double x, double min, double max)
{
    /*
     * Red = 0
     * Green = 1
     * Blue = 2
     */
    double posSlope = (max - min) / 60;
    double negSlope = (min - max) / 60;

    if (x < 60)
    {
        color[0] = max;
        color[1] = posSlope * x + min;
        color[2] = min;
        return;
    }
    else if (x < 120)
    {
        color[0] = negSlope * x + 2.0 * max + min;
        color[1] = max;
        color[2] = min;
        return;
    }
    else if (x < 180)
    {
        color[0] = min;
        color[1] = max;
        color[2] = posSlope * x - 2.0 * max + min;
        return;
    }
    else if (x < 240)
    {
        color[0] = min;
        color[1] = negSlope * x + 4.0 * max + min;
        color[2] = max;
        return;
    }
    else if (x < 300)
    {
        color[0] = posSlope * x - 4.0 * max + min;
        color[1] = min;
        color[2] = max;
        return;
    }
    else
    {
        color[0] = max;
        color[1] = min;
        color[2] = negSlope * x + 6 * max;
        return;
    }
}


/**
 * @brief Get the GlobalIdx 1D 1D object
 * Obtained this function from 'https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf'
 * @return __device__ 
 */
__device__
int getGlobalIdx_1D_1D(){
    return blockIdx.x * blockDim.x + threadIdx.x;
}


/**
 * @brief mandelbrot is a global function for CUDA device, where it will accept the mandelbrot
 * grid, and check whether the provided number is part of the mandelbrot set, or not. If it is not
 * it will assign the iteration number for later colour grading.
 * 
 * @param mandelGrid - mandelbrot grid matrix structure storing properties and elements
 * @return __global__ 
 */
__global__ void mandelbrot(Matrix mandelGrid)
{
    // Assign pixel's ID in global grid
    int pixel = getGlobalIdx_1D_1D();

    // Check if pixel is within total pixels
    if ((pixel < mandelGrid.total_pixels))
    {
        // Assign row and column of hte pixel
        int row = pixel / mandelGrid.width;
        int column = pixel % mandelGrid.width;

        // Work out offset amount
        int xoffset = -(mandelGrid.width - 1) / 2;
        int yoffset = (mandelGrid.height - 1) / 2;

        // Get cartesian grid x and y with offset and zoom applied
        double x  = mandelGrid.xcenter + (double)(xoffset + column) / mandelGrid.zoom;
        double y = mandelGrid.ycenter + (double)(yoffset - row) / mandelGrid.zoom;

        // Work out whether x and y is within mandelbrot set, and store iteration element
        double a = 0;
        double b = 0;
        double aold = 0;
        double bold = 0;
        double zmagsqr = 0;
        int iter = 0;
        while (iter < MAX_ITER && zmagsqr <= 4.0)
        {
            ++iter;
            a = (aold * aold) - (bold * bold) + x;
            b = 2.0 * aold * bold + y;

            zmagsqr = a * a + b * b;

            aold = a;
            bold = b;
        }

        mandelGrid.elements[pixel] = iter;
    }
    
    
}

int main(int argc, char **argv)
{
    printf("[Mandelbrot set calculation using CUDA] - Starting...\n");

    // Check whether user requested help
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
        checkCmdLineFlag(argc, (const char **)argv, "?"))
    {
        printf("Usage: -device=n (where deviceId >= 0, and default is 0) \n");
        printf("       -z=Zoom (The precision of the mandelbrot set) \n");
        printf("       -w=Width -h=Height (Pixel Width x Pixel Height of Bitmap)\n");
        printf("       -x=X -y=Y (X and Y center of bitmap for mandelbrot set)\n");
        printf("  Note: Zoom, Width and Height must all be positive non zero integers\n");

        exit(EXIT_SUCCESS);
    }

    // Check whether user input device ID other set as default
    int devID = 0;
    if (checkCmdLineFlag(argc, (const char **) argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char**)argv, "device");
        cudaSetDevice(devID);
    }

    // Setup error and device properties check for CUDA calls
    cudaError_t err = cudaSuccess;
    cudaDeviceProp deviceProp;
    err = cudaGetDevice(&devID);
    if (err != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
    }
    err = cudaGetDeviceProperties(&deviceProp, devID);

    // Check whether device can be computed on
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }

    //Print device properties
    if (err != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(err), err, __LINE__);
    }
    else
    {
        printf("\nGpu Device:\n");
        printf("ID %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Create host mandelGrid with defaults
    Matrix mandelGrid;
    mandelGrid.width = WIDTH;
    mandelGrid.height = HEIGHT;
    mandelGrid.xcenter = XCENTER; // double
    mandelGrid.ycenter = YCENTER; // double
    mandelGrid.zoom = ZOOM; //double

    // Override defaults with user input on command line
    if (checkCmdLineFlag(argc, (const char **)argv, "w"))
    {
        mandelGrid.width = getCmdLineArgumentInt(argc, (const char **)argv, "w");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "h"))
    {
        mandelGrid.height = getCmdLineArgumentInt(argc, (const char **)argv, "h");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "z"))
    {   
        
        mandelGrid.zoom = (double)getCmdLineArgumentFloat(argc, (const char **)argv, "z");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "x"))
    {
        mandelGrid.xcenter = (double)getCmdLineArgumentFloat(argc, (const char **)argv, "x");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "y"))
    {
        mandelGrid.ycenter = (double)getCmdLineArgumentFloat(argc, (const char **)argv, "y");
    }

    // Get total pixels
    mandelGrid.total_pixels = mandelGrid.width * mandelGrid.height;

    printf( "Mandelbrot set will have:\n\tWidth: %i\n\tHeight: %i\n\tPixels: %i\n\tX Center: %.3f\n\tY Center: %.3f\n\tZoom: %.3f\n",
            mandelGrid.width,
            mandelGrid.height,
            mandelGrid.total_pixels,
            mandelGrid.xcenter,
            mandelGrid.ycenter,
            mandelGrid.zoom);
    printf("\t(Note: X Center, Y Center and Zoom rounded to 3 decimal\n\tfor this print only.)\n\n");

    // Allocate memory for host mandelGrid elements
    mandelGrid.elements = (int *)malloc(mandelGrid.total_pixels * sizeof(int*));

    // Create  device mandelGrid
    Matrix d_mandelGrid;
    d_mandelGrid.width = mandelGrid.width;
    d_mandelGrid.height = mandelGrid.height;
    d_mandelGrid.zoom = mandelGrid.zoom;
    d_mandelGrid.xcenter = mandelGrid.xcenter;
    d_mandelGrid.ycenter = mandelGrid.ycenter;
    d_mandelGrid.total_pixels = mandelGrid.width * mandelGrid.height;

    // Allocate memory for device mandelGrid elements
    size_t size = d_mandelGrid.total_pixels * sizeof(int*);
    err = cudaMalloc(&d_mandelGrid.elements, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory for original grid (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Calculate blocks needed
    int blocks, threads;
    threads = 1023;
    blocks = (mandelGrid.total_pixels / threads) + 1;

    fprintf(stderr,"CUDA will use %i blocks, which will contain %i threads each.\n\n",blocks, threads);

    cudaDeviceSynchronize();

    // Time the CUDA event for performance
    cudaEvent_t start;
    err = cudaEventCreate(&start);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEvent_t stop;
    err = cudaEventCreate(&stop);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Starting Mandelbrot calculation on CUDA device\n\n");

    // Record the start event
    err = cudaEventRecord(start, NULL);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Start CUDA kernel for calculating mandel grid
    mandelbrot<<<blocks, threads>>>(d_mandelGrid);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernal (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Record the stop event
    err = cudaEventRecord(stop, NULL);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    err = cudaEventSynchronize(stop);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Mandelbrot set calculation on CUDA device complete.\n\n");

    float msecTotal = 0.0f;
    err = cudaEventElapsedTime(&msecTotal, start, stop);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Compute and print the performance
    float msecPerBlock = msecTotal / (float)blocks ;
    float msecPerThread = msecTotal / (float)blocks / (float)threads;
    printf("CUDA performance:\n");
    printf(
        "Time: %.2f mSec Total, %.6f mSec Per Block, %.6f mSec Per Thread, \nPixels: %i, \nWorkgroupSize: %i blocks, %i threads\n\n",
        msecTotal,
        msecPerBlock,
        msecPerThread,
        mandelGrid.width * mandelGrid.height,
        blocks, 
        threads);


    err = cudaMemcpy(mandelGrid.elements, d_mandelGrid.elements, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy device results back to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Create bitmap
    printf("Creating Bitmap\n\n");

    bmpfile_t *bmp;
    bmp = bmp_create(mandelGrid.width, mandelGrid.height, 32);
    rgb_pixel_t pixel = {0, 0, 0, 0};
    
    for (int pixelxy = 0; pixelxy < mandelGrid.total_pixels; pixelxy++)
    {

        int row = pixelxy / mandelGrid.width;
        int column = pixelxy % mandelGrid.width;

        double color[3];
        double x_col = (COLOUR_MAX - ((((float)mandelGrid.elements[pixelxy] / ((float)MAX_ITER) * GRADIENT_COLOUR_MAX))));
        
        GroundColorMix(color, x_col, 1, COLOUR_DEPTH);
        
        pixel.red = color[0];
        pixel.green = color[1];
        pixel.blue = color[2];
        
        bmp_set_pixel(bmp, column, row, pixel);

        
    }

    printf("Finished. Free'ing memory, reseting CUDA and closing\n");

    bmp_save(bmp, FILENAME);
    bmp_destroy(bmp);

    free(mandelGrid.elements);
    cudaFree(d_mandelGrid.elements);

    cudaDeviceReset();

    
    return 0;
      
}
