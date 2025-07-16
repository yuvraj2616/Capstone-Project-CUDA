#include "benchmark.h"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>

Benchmark::Benchmark() {
}

Benchmark::~Benchmark() {
}

void Benchmark::startTimer() {
    start_time = std::chrono::high_resolution_clock::now();
}

double Benchmark::stopTimer() {
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    return duration.count() / 1000.0; // Convert to milliseconds
}

PerformanceMetrics Benchmark::analyzeResults(const std::vector<double>& times,
                                              const std::string& operation_name,
                                              size_t image_size_pixels) {
    PerformanceMetrics metrics;
    metrics.operation_name = operation_name;
    metrics.iterations = times.size();
    metrics.image_size_pixels = image_size_pixels;
    
    if (times.empty()) {
        throw std::runtime_error("No timing data provided for analysis");
    }
    
    metrics.min_time_ms = calculateMin(times);
    metrics.max_time_ms = calculateMax(times);
    metrics.avg_time_ms = calculateMean(times);
    metrics.std_dev_ms = calculateStdDev(times, metrics.avg_time_ms);
    
    // Calculate throughput in megapixels per second
    metrics.throughput_mpixels_per_sec = (image_size_pixels / 1000000.0) / (metrics.avg_time_ms / 1000.0);
    
    return metrics;
}

void Benchmark::printSummary(const std::vector<PerformanceMetrics>& results) {
    std::cout << "Detailed Performance Analysis\n";
    std::cout << "============================\n\n";
    
    for (const auto& result : results) {
        std::cout << "Operation: " << result.operation_name << std::endl;
        std::cout << "  Image Size: " << result.image_size_pixels << " pixels" << std::endl;
        std::cout << "  Iterations: " << result.iterations << std::endl;
        std::cout << "  Average Time: " << std::fixed << std::setprecision(3) 
                  << result.avg_time_ms << " ms" << std::endl;
        std::cout << "  Min Time: " << std::fixed << std::setprecision(3) 
                  << result.min_time_ms << " ms" << std::endl;
        std::cout << "  Max Time: " << std::fixed << std::setprecision(3) 
                  << result.max_time_ms << " ms" << std::endl;
        std::cout << "  Std Deviation: " << std::fixed << std::setprecision(3) 
                  << result.std_dev_ms << " ms" << std::endl;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2) 
                  << result.throughput_mpixels_per_sec << " MP/s" << std::endl;
        std::cout << std::endl;
    }
}

void Benchmark::saveResultsToCSV(const std::vector<PerformanceMetrics>& results,
                                  const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    // Write CSV header
    file << "Operation,Image_Size_Pixels,Iterations,Avg_Time_ms,Min_Time_ms,Max_Time_ms,Std_Dev_ms,Throughput_MPps\n";
    
    // Write data rows
    for (const auto& result : results) {
        file << result.operation_name << ","
             << result.image_size_pixels << ","
             << result.iterations << ","
             << std::fixed << std::setprecision(6) << result.avg_time_ms << ","
             << std::fixed << std::setprecision(6) << result.min_time_ms << ","
             << std::fixed << std::setprecision(6) << result.max_time_ms << ","
             << std::fixed << std::setprecision(6) << result.std_dev_ms << ","
             << std::fixed << std::setprecision(6) << result.throughput_mpixels_per_sec << "\n";
    }
    
    file.close();
}

void Benchmark::printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    std::cout << "CUDA Device Information\n";
    std::cout << "======================\n";
    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    
    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found!" << std::endl;
        return;
    }
    
    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        std::cout << "\nDevice " << device << ": " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Grid Size: " << prop.maxGridSize[0] << " x " 
                  << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
        std::cout << "  Max Block Dimensions: " << prop.maxThreadsDim[0] << " x " 
                  << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate / 1000 << " MHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  Peak Memory Bandwidth: " 
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 
                  << " GB/s" << std::endl;
    }
}

void Benchmark::checkCUDAError(const std::string& operation_name) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(operation_name + " failed: " + std::string(cudaGetErrorString(error)));
    }
}

// Statistical calculation helpers
double Benchmark::calculateMean(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double Benchmark::calculateStdDev(const std::vector<double>& values, double mean) {
    if (values.size() <= 1) return 0.0;
    
    double sum_squared_diff = 0.0;
    for (double value : values) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }
    
    return std::sqrt(sum_squared_diff / (values.size() - 1));
}

double Benchmark::calculateMin(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return *std::min_element(values.begin(), values.end());
}

double Benchmark::calculateMax(const std::vector<double>& values) {
    if (values.empty()) return 0.0;
    return *std::max_element(values.begin(), values.end());
}

// Global benchmark utilities
void warmupGPU() {
    // Allocate and free some memory to warm up the GPU
    void* d_ptr;
    cudaMalloc(&d_ptr, 1024 * 1024); // 1MB
    cudaFree(d_ptr);
    
    // Run a simple kernel to ensure GPU is ready
    cudaDeviceSynchronize();
}

void synchronizeGPU() {
    cudaDeviceSynchronize();
}
