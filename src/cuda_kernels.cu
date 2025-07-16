#include "cuda_kernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 15

// Shared memory for convolution optimization
__shared__ unsigned char shared_mem[BLOCK_SIZE + MAX_KERNEL_SIZE - 1][BLOCK_SIZE + MAX_KERNEL_SIZE - 1][3];

// Device constant memory for Gaussian kernel
__constant__ float d_gaussian_kernel[MAX_KERNEL_SIZE];

// 2D Gaussian Blur Kernel with shared memory optimization
__global__ void gaussianBlurKernel(unsigned char* input, 
                                   unsigned char* output,
                                   int width, int height, int channels,
                                   float* kernel, int kernel_size) {
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    
    int half_kernel = kernel_size / 2;
    
    // Load data into shared memory with halo
    for (int dy = -half_kernel; dy <= half_kernel; dy++) {
        for (int dx = -half_kernel; dx <= half_kernel; dx++) {
            int shared_x = tx + dx + half_kernel;
            int shared_y = ty + dy + half_kernel;
            int global_x = x + dx;
            int global_y = y + dy;
            
            // Boundary handling with clamp
            global_x = max(0, min(width - 1, global_x));
            global_y = max(0, min(height - 1, global_y));
            
            if (shared_x < BLOCK_SIZE + MAX_KERNEL_SIZE - 1 && 
                shared_y < BLOCK_SIZE + MAX_KERNEL_SIZE - 1) {
                for (int c = 0; c < channels; c++) {
                    shared_mem[shared_y][shared_x][c] = 
                        input[(global_y * width + global_x) * channels + c];
                }
            }
        }
    }
    
    __syncthreads();
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int ky = 0; ky < kernel_size; ky++) {
                for (int kx = 0; kx < kernel_size; kx++) {
                    int shared_x = tx + kx;
                    int shared_y = ty + ky;
                    float kernel_val = kernel[ky * kernel_size + kx];
                    sum += shared_mem[shared_y][shared_x][c] * kernel_val;
                }
            }
            
            output[(y * width + x) * channels + c] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum));
        }
    }
}

// Optimized separable Gaussian blur - Horizontal pass
__global__ void gaussianBlurHorizontalKernel(unsigned char* input,
                                              float* temp,
                                              int width, int height, int channels,
                                              float* kernel, int kernel_size) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half_kernel = kernel_size / 2;
        
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int sample_x = x + k - half_kernel;
                sample_x = max(0, min(width - 1, sample_x)); // Clamp to boundaries
                
                sum += input[(y * width + sample_x) * channels + c] * kernel[k];
            }
            
            temp[(y * width + x) * channels + c] = sum;
        }
    }
}

// Optimized separable Gaussian blur - Vertical pass
__global__ void gaussianBlurVerticalKernel(float* temp,
                                           unsigned char* output,
                                           int width, int height, int channels,
                                           float* kernel, int kernel_size) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int half_kernel = kernel_size / 2;
        
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            
            for (int k = 0; k < kernel_size; k++) {
                int sample_y = y + k - half_kernel;
                sample_y = max(0, min(height - 1, sample_y)); // Clamp to boundaries
                
                sum += temp[(sample_y * width + x) * channels + c] * kernel[k];
            }
            
            output[(y * width + x) * channels + c] = (unsigned char)fminf(255.0f, fmaxf(0.0f, sum));
        }
    }
}

// Sobel Edge Detection Kernel
__global__ void sobelEdgeDetectionKernel(unsigned char* input,
                                         unsigned char* output,
                                         int width, int height, int channels) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        // Sobel Y kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        
        for (int c = 0; c < channels; c++) {
            float gx = 0.0f, gy = 0.0f;
            
            // Apply Sobel X kernel
            gx += -1 * input[((y-1) * width + (x-1)) * channels + c];
            gx +=  1 * input[((y-1) * width + (x+1)) * channels + c];
            gx += -2 * input[(y * width + (x-1)) * channels + c];
            gx +=  2 * input[(y * width + (x+1)) * channels + c];
            gx += -1 * input[((y+1) * width + (x-1)) * channels + c];
            gx +=  1 * input[((y+1) * width + (x+1)) * channels + c];
            
            // Apply Sobel Y kernel
            gy += -1 * input[((y-1) * width + (x-1)) * channels + c];
            gy += -2 * input[((y-1) * width + x) * channels + c];
            gy += -1 * input[((y-1) * width + (x+1)) * channels + c];
            gy +=  1 * input[((y+1) * width + (x-1)) * channels + c];
            gy +=  2 * input[((y+1) * width + x) * channels + c];
            gy +=  1 * input[((y+1) * width + (x+1)) * channels + c];
            
            // Calculate gradient magnitude
            float magnitude = sqrtf(gx * gx + gy * gy);
            output[(y * width + x) * channels + c] = (unsigned char)fminf(255.0f, magnitude);
        }
    } else {
        // Set border pixels to 0
        for (int c = 0; c < channels; c++) {
            output[(y * width + x) * channels + c] = 0;
        }
    }
}

// RGB to Grayscale Conversion Kernel
__global__ void rgbToGrayscaleKernel(unsigned char* input,
                                     unsigned char* output,
                                     int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x < width && y < height) {
        unsigned char r = input[idx * 3 + 0];
        unsigned char g = input[idx * 3 + 1];
        unsigned char b = input[idx * 3 + 2];
        
        // Standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        output[idx] = (unsigned char)fminf(255.0f, fmaxf(0.0f, gray));
    }
}

// RGB to HSV Conversion Kernel
__global__ void rgbToHSVKernel(unsigned char* input,
                               unsigned char* output,
                               int width, int height) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x < width && y < height) {
        float r = input[idx * 3 + 0] / 255.0f;
        float g = input[idx * 3 + 1] / 255.0f;
        float b = input[idx * 3 + 2] / 255.0f;
        
        float max_val = fmaxf(r, fmaxf(g, b));
        float min_val = fminf(r, fminf(g, b));
        float delta = max_val - min_val;
        
        float h = 0.0f, s = 0.0f, v = max_val;
        
        if (delta > 0.0f) {
            s = delta / max_val;
            
            if (max_val == r) {
                h = 60.0f * (fmodf(((g - b) / delta), 6.0f));
            } else if (max_val == g) {
                h = 60.0f * (((b - r) / delta) + 2.0f);
            } else {
                h = 60.0f * (((r - g) / delta) + 4.0f);
            }
            
            if (h < 0.0f) h += 360.0f;
        }
        
        output[idx * 3 + 0] = (unsigned char)(h * 255.0f / 360.0f);
        output[idx * 3 + 1] = (unsigned char)(s * 255.0f);
        output[idx * 3 + 2] = (unsigned char)(v * 255.0f);
    }
}

// Normalization kernel for float to unsigned char conversion
__global__ void normalizeKernel(float* input, unsigned char* output,
                                int width, int height, int channels,
                                float min_val, float max_val) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            float val = input[idx * channels + c];
            float normalized = (val - min_val) / (max_val - min_val) * 255.0f;
            output[idx * channels + c] = (unsigned char)fminf(255.0f, fmaxf(0.0f, normalized));
        }
    }
}

// Host wrapper functions
cudaError_t launchGaussianBlur(unsigned char* d_input, unsigned char* d_output,
                               float* d_temp, int width, int height, int channels,
                               float sigma) {
    
    // Generate Gaussian kernel
    int kernel_size = calculateKernelSize(sigma);
    float* h_kernel = new float[kernel_size];
    generateGaussianKernel(h_kernel, kernel_size, sigma);
    
    // Copy kernel to device
    float* d_kernel;
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch separable convolution kernels
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Horizontal pass
    gaussianBlurHorizontalKernel<<<gridSize, blockSize>>>(
        d_input, d_temp, width, height, channels, d_kernel, kernel_size);
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        delete[] h_kernel;
        cudaFree(d_kernel);
        return error;
    }
    
    // Vertical pass
    gaussianBlurVerticalKernel<<<gridSize, blockSize>>>(
        d_temp, d_output, width, height, channels, d_kernel, kernel_size);
    
    error = cudaGetLastError();
    
    // Cleanup
    delete[] h_kernel;
    cudaFree(d_kernel);
    
    return error;
}

cudaError_t launchSobelEdgeDetection(unsigned char* d_input, unsigned char* d_output,
                                     int width, int height, int channels) {
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    sobelEdgeDetectionKernel<<<gridSize, blockSize>>>(
        d_input, d_output, width, height, channels);
    
    return cudaGetLastError();
}

cudaError_t launchRGBToGrayscale(unsigned char* d_input, unsigned char* d_output,
                                 int width, int height) {
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rgbToGrayscaleKernel<<<gridSize, blockSize>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError();
}

cudaError_t launchRGBToHSV(unsigned char* d_input, unsigned char* d_output,
                           int width, int height) {
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    rgbToHSVKernel<<<gridSize, blockSize>>>(
        d_input, d_output, width, height);
    
    return cudaGetLastError();
}

// Memory management utilities
cudaError_t allocateGPUMemory(unsigned char** d_ptr, size_t size) {
    return cudaMalloc(d_ptr, size);
}

cudaError_t deallocateGPUMemory(unsigned char* d_ptr) {
    return cudaFree(d_ptr);
}

// Gaussian kernel generation
void generateGaussianKernel(float* kernel, int size, float sigma) {
    int half = size / 2;
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        int x = i - half;
        kernel[i] = expf(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < size; i++) {
        kernel[i] /= sum;
    }
}

int calculateKernelSize(float sigma) {
    // Rule of thumb: kernel size should be at least 6*sigma, and odd
    int size = (int)ceilf(6.0f * sigma);
    if (size % 2 == 0) size++;
    return min(size, MAX_KERNEL_SIZE);
}
