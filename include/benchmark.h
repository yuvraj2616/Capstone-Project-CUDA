#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <iomanip>

struct PerformanceMetrics {
    std::string operation_name;
    double min_time_ms;
    double max_time_ms;
    double avg_time_ms;
    double std_dev_ms;
    size_t iterations;
    size_t image_size_pixels;
    double throughput_mpixels_per_sec;
};

class Benchmark {
public:
    Benchmark();
    ~Benchmark();

    // Timing utilities
    void startTimer();
    double stopTimer(); // Returns elapsed time in milliseconds
    
    // Statistical analysis
    PerformanceMetrics analyzeResults(const std::vector<double>& times,
                                      const std::string& operation_name,
                                      size_t image_size_pixels);
    
    // Results reporting
    void printSummary(const std::vector<PerformanceMetrics>& results);
    void saveResultsToCSV(const std::vector<PerformanceMetrics>& results,
                          const std::string& filename);
    
    // GPU profiling utilities
    void printGPUInfo();
    void checkCUDAError(const std::string& operation_name);

private:
    std::chrono::high_resolution_clock::time_point start_time;
    
    // Statistical calculation helpers
    double calculateMean(const std::vector<double>& values);
    double calculateStdDev(const std::vector<double>& values, double mean);
    double calculateMin(const std::vector<double>& values);
    double calculateMax(const std::vector<double>& values);
};

// Global benchmark utilities
void warmupGPU();
void synchronizeGPU();

#endif // BENCHMARK_H
