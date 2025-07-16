#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernel declarations
extern "C" {
    
// Convolution kernel for Gaussian blur
__global__ void gaussianBlurKernel(unsigned char* input, 
                                   unsigned char* output,
                                   int width, int height, int channels,
                                   float* kernel, int kernel_size);

// Separable Gaussian blur kernels for better performance
__global__ void gaussianBlurHorizontalKernel(unsigned char* input,
                                              float* temp,
                                              int width, int height, int channels,
                                              float* kernel, int kernel_size);

__global__ void gaussianBlurVerticalKernel(float* temp,
                                           unsigned char* output,
                                           int width, int height, int channels,
                                           float* kernel, int kernel_size);

// Sobel edge detection kernel
__global__ void sobelEdgeDetectionKernel(unsigned char* input,
                                         unsigned char* output,
                                         int width, int height, int channels);

// RGB to Grayscale conversion kernel
__global__ void rgbToGrayscaleKernel(unsigned char* input,
                                     unsigned char* output,
                                     int width, int height);

// RGB to HSV conversion kernel
__global__ void rgbToHSVKernel(unsigned char* input,
                               unsigned char* output,
                               int width, int height);

// Utility kernels
__global__ void normalizeKernel(float* input, unsigned char* output,
                                int width, int height, int channels,
                                float min_val, float max_val);

}

// Host wrapper functions
cudaError_t launchGaussianBlur(unsigned char* d_input, unsigned char* d_output,
                               float* d_temp, int width, int height, int channels,
                               float sigma);

cudaError_t launchSobelEdgeDetection(unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, int channels);

cudaError_t launchRGBToGrayscale(unsigned char* d_input, unsigned char* d_output,
                                 int width, int height);

cudaError_t launchRGBToHSV(unsigned char* d_input, unsigned char* d_output,
                           int width, int height);

// Memory management utilities
cudaError_t allocateGPUMemory(unsigned char** d_ptr, size_t size);
cudaError_t deallocateGPUMemory(unsigned char* d_ptr);

// Kernel generation utilities
void generateGaussianKernel(float* kernel, int size, float sigma);
int calculateKernelSize(float sigma);

#endif // CUDA_KERNELS_CUH
