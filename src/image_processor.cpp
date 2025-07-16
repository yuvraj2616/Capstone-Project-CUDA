#include "image_processor.h"
#include "cuda_kernels.cuh"
#include "benchmark.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

ImageProcessor::ImageProcessor() 
    : d_input(nullptr), d_output(nullptr), d_temp(nullptr), 
      allocated_size(0), gpu_memory_allocated(false) {
}

ImageProcessor::~ImageProcessor() {
    deallocateGPUMemory();
}

void ImageProcessor::allocateGPUMemory(size_t width, size_t height, size_t channels) {
    size_t required_size = width * height * channels * sizeof(unsigned char);
    size_t temp_size = width * height * channels * sizeof(float);
    
    if (!gpu_memory_allocated || allocated_size < required_size) {
        // Deallocate existing memory
        deallocateGPUMemory();
        
        // Allocate new memory with some padding for efficiency
        allocated_size = required_size * 1.2; // 20% padding
        
        cudaError_t error;
        error = cudaMalloc(&d_input, allocated_size);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to allocate GPU input memory: " + 
                                   std::string(cudaGetErrorString(error)));
        }
        
        error = cudaMalloc(&d_output, allocated_size);
        if (error != cudaSuccess) {
            cudaFree(d_input);
            throw std::runtime_error("Failed to allocate GPU output memory: " + 
                                   std::string(cudaGetErrorString(error)));
        }
        
        error = cudaMalloc(&d_temp, temp_size);
        if (error != cudaSuccess) {
            cudaFree(d_input);
            cudaFree(d_output);
            throw std::runtime_error("Failed to allocate GPU temp memory: " + 
                                   std::string(cudaGetErrorString(error)));
        }
        
        gpu_memory_allocated = true;
    }
}

void ImageProcessor::deallocateGPUMemory() {
    if (gpu_memory_allocated) {
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_temp) cudaFree(d_temp);
        
        d_input = d_output = nullptr;
        d_temp = nullptr;
        allocated_size = 0;
        gpu_memory_allocated = false;
    }
}

void ImageProcessor::copyToGPU(const cv::Mat& input) {
    size_t size = input.total() * input.elemSize();
    cudaError_t error = cudaMemcpy(d_input, input.ptr(), size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy data to GPU: " + 
                               std::string(cudaGetErrorString(error)));
    }
}

cv::Mat ImageProcessor::copyFromGPU(size_t width, size_t height, size_t channels) {
    cv::Mat result;
    if (channels == 1) {
        result = cv::Mat(height, width, CV_8UC1);
    } else if (channels == 3) {
        result = cv::Mat(height, width, CV_8UC3);
    } else {
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(channels));
    }
    
    size_t size = result.total() * result.elemSize();
    cudaError_t error = cudaMemcpy(result.ptr(), d_output, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from GPU: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    return result;
}

void ImageProcessor::startTimer() {
    start_time = std::chrono::high_resolution_clock::now();
}

double ImageProcessor::stopTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// CPU implementations for comparison
cv::Mat ImageProcessor::gaussianBlurCPU(const cv::Mat& input, float sigma) {
    cv::Mat result;
    int kernel_size = calculateKernelSize(sigma);
    cv::GaussianBlur(input, result, cv::Size(kernel_size, kernel_size), sigma);
    return result;
}

cv::Mat ImageProcessor::sobelEdgeDetectionCPU(const cv::Mat& input) {
    cv::Mat gray, grad_x, grad_y, result;
    
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
    
    cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
    
    cv::convertScaleAbs(grad_x, grad_x);
    cv::convertScaleAbs(grad_y, grad_y);
    cv::addWeighted(grad_x, 0.5, grad_y, 0.5, 0, result);
    
    if (input.channels() == 3) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    }
    
    return result;
}

cv::Mat ImageProcessor::rgbToGrayscaleCPU(const cv::Mat& input) {
    cv::Mat result;
    if (input.channels() == 3) {
        cv::cvtColor(input, result, cv::COLOR_BGR2GRAY);
    } else {
        result = input.clone();
    }
    return result;
}

cv::Mat ImageProcessor::rgbToHSVCPU(const cv::Mat& input) {
    cv::Mat result;
    cv::cvtColor(input, result, cv::COLOR_BGR2HSV);
    return result;
}

// GPU implementations
cv::Mat ImageProcessor::gaussianBlurGPU(const cv::Mat& input, float sigma) {
    allocateGPUMemory(input.cols, input.rows, input.channels());
    copyToGPU(input);
    
    cudaError_t error = launchGaussianBlur(d_input, d_output, d_temp,
                                           input.cols, input.rows, input.channels(), sigma);
    
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA Gaussian blur failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    cudaDeviceSynchronize();
    return copyFromGPU(input.cols, input.rows, input.channels());
}

cv::Mat ImageProcessor::sobelEdgeDetectionGPU(const cv::Mat& input) {
    allocateGPUMemory(input.cols, input.rows, input.channels());
    copyToGPU(input);
    
    cudaError_t error = launchSobelEdgeDetection(d_input, d_output,
                                                 input.cols, input.rows, input.channels());
    
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA Sobel edge detection failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    cudaDeviceSynchronize();
    return copyFromGPU(input.cols, input.rows, input.channels());
}

cv::Mat ImageProcessor::rgbToGrayscaleGPU(const cv::Mat& input) {
    if (input.channels() != 3) {
        return input.clone(); // Already grayscale
    }
    
    allocateGPUMemory(input.cols, input.rows, 3); // Input has 3 channels
    copyToGPU(input);
    
    cudaError_t error = launchRGBToGrayscale(d_input, d_output,
                                             input.cols, input.rows);
    
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA RGB to Grayscale failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    cudaDeviceSynchronize();
    return copyFromGPU(input.cols, input.rows, 1); // Output has 1 channel
}

cv::Mat ImageProcessor::rgbToHSVGPU(const cv::Mat& input) {
    if (input.channels() != 3) {
        throw std::runtime_error("RGB to HSV requires 3-channel input");
    }
    
    allocateGPUMemory(input.cols, input.rows, 3);
    copyToGPU(input);
    
    cudaError_t error = launchRGBToHSV(d_input, d_output,
                                       input.cols, input.rows);
    
    if (error != cudaSuccess) {
        throw std::runtime_error("CUDA RGB to HSV failed: " + 
                               std::string(cudaGetErrorString(error)));
    }
    
    cudaDeviceSynchronize();
    return copyFromGPU(input.cols, input.rows, 3);
}

std::vector<cv::Mat> ImageProcessor::processBatch(const std::vector<cv::Mat>& images,
                                                   const std::string& filter_type,
                                                   float sigma) {
    std::vector<cv::Mat> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        cv::Mat result;
        
        if (filter_type == "gaussian") {
            result = gaussianBlurGPU(image, sigma);
        }
        else if (filter_type == "sobel") {
            result = sobelEdgeDetectionGPU(image);
        }
        else if (filter_type == "grayscale") {
            result = rgbToGrayscaleGPU(image);
        }
        else if (filter_type == "hsv") {
            result = rgbToHSVGPU(image);
        }
        else {
            throw std::runtime_error("Unknown filter type: " + filter_type);
        }
        
        results.push_back(result);
    }
    
    return results;
}

BenchmarkResult ImageProcessor::benchmark(const cv::Mat& input,
                                          const std::string& operation,
                                          int iterations) {
    BenchmarkResult result;
    result.operation = operation;
    result.image_width = input.cols;
    result.image_height = input.rows;
    
    // Warm up
    for (int i = 0; i < 5; i++) {
        if (operation == "gaussian") {
            gaussianBlurGPU(input, 2.0f);
        }
    }
    
    // Benchmark CPU
    std::vector<double> cpu_times;
    for (int i = 0; i < iterations; i++) {
        startTimer();
        
        if (operation == "gaussian") {
            gaussianBlurCPU(input, 2.0f);
        }
        else if (operation == "sobel") {
            sobelEdgeDetectionCPU(input);
        }
        else if (operation == "grayscale") {
            rgbToGrayscaleCPU(input);
        }
        else if (operation == "hsv") {
            rgbToHSVCPU(input);
        }
        
        cpu_times.push_back(stopTimer());
    }
    
    // Benchmark GPU
    std::vector<double> gpu_times;
    for (int i = 0; i < iterations; i++) {
        startTimer();
        
        if (operation == "gaussian") {
            gaussianBlurGPU(input, 2.0f);
        }
        else if (operation == "sobel") {
            sobelEdgeDetectionGPU(input);
        }
        else if (operation == "grayscale") {
            rgbToGrayscaleGPU(input);
        }
        else if (operation == "hsv") {
            rgbToHSVGPU(input);
        }
        
        gpu_times.push_back(stopTimer());
    }
    
    // Calculate averages
    result.cpu_time_ms = std::accumulate(cpu_times.begin(), cpu_times.end(), 0.0) / iterations;
    result.gpu_time_ms = std::accumulate(gpu_times.begin(), gpu_times.end(), 0.0) / iterations;
    result.speedup = result.cpu_time_ms / result.gpu_time_ms;
    
    return result;
}

void ImageProcessor::printBenchmarkResults(const std::vector<BenchmarkResult>& results) {
    std::cout << "Performance Benchmark Results\n";
    std::cout << "============================\n\n";
    std::cout << std::setw(12) << "Operation" << std::setw(15) << "CPU Time (ms)" 
              << std::setw(15) << "GPU Time (ms)" << std::setw(12) << "Speedup" << std::endl;
    std::cout << std::string(54, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(12) << result.operation
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.cpu_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << result.gpu_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(2) << result.speedup << "x"
                  << std::endl;
    }
    std::cout << std::endl;
}

// Utility functions
std::vector<cv::Mat> loadImagesFromDirectory(const std::string& directory) {
    std::vector<cv::Mat> images;
    std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
    
    try {
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                
                if (std::find(extensions.begin(), extensions.end(), ext) != extensions.end()) {
                    cv::Mat image = cv::imread(entry.path().string());
                    if (!image.empty()) {
                        images.push_back(image);
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
    
    return images;
}

void saveImagesToDirectory(const std::vector<cv::Mat>& images,
                          const std::string& directory,
                          const std::string& prefix) {
    fs::create_directories(directory);
    
    for (size_t i = 0; i < images.size(); i++) {
        std::string filename = prefix + std::to_string(i) + ".png";
        std::string filepath = (fs::path(directory) / filename).string();
        cv::imwrite(filepath, images[i]);
    }
}

cv::Mat loadImage(const std::string& filepath) {
    cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Warning: Could not load image " << filepath << std::endl;
    }
    return image;
}

void saveImage(const cv::Mat& image, const std::string& filepath) {
    if (!cv::imwrite(filepath, image)) {
        throw std::runtime_error("Failed to save image to " + filepath);
    }
}
