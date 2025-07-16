#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>

struct BenchmarkResult {
    std::string operation;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    size_t image_width;
    size_t image_height;
};

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // CPU implementations for comparison
    cv::Mat gaussianBlurCPU(const cv::Mat& input, float sigma);
    cv::Mat sobelEdgeDetectionCPU(const cv::Mat& input);
    cv::Mat rgbToGrayscaleCPU(const cv::Mat& input);
    cv::Mat rgbToHSVCPU(const cv::Mat& input);

    // GPU implementations
    cv::Mat gaussianBlurGPU(const cv::Mat& input, float sigma);
    cv::Mat sobelEdgeDetectionGPU(const cv::Mat& input);
    cv::Mat rgbToGrayscaleGPU(const cv::Mat& input);
    cv::Mat rgbToHSVGPU(const cv::Mat& input);

    // Batch processing
    std::vector<cv::Mat> processBatch(const std::vector<cv::Mat>& images, 
                                      const std::string& filter_type,
                                      float sigma = 1.0f);

    // Benchmarking
    BenchmarkResult benchmark(const cv::Mat& input, 
                              const std::string& operation,
                              int iterations = 100);
    
    void printBenchmarkResults(const std::vector<BenchmarkResult>& results);

private:
    // GPU memory management
    void allocateGPUMemory(size_t width, size_t height, size_t channels);
    void deallocateGPUMemory();
    void copyToGPU(const cv::Mat& input);
    cv::Mat copyFromGPU(size_t width, size_t height, size_t channels);

    // GPU memory pointers
    unsigned char* d_input;
    unsigned char* d_output;
    float* d_temp;
    size_t allocated_size;
    bool gpu_memory_allocated;

    // Timing utilities
    std::chrono::high_resolution_clock::time_point start_time;
    void startTimer();
    double stopTimer(); // returns elapsed time in milliseconds
};

// Utility functions
std::vector<cv::Mat> loadImagesFromDirectory(const std::string& directory);
void saveImagesToDirectory(const std::vector<cv::Mat>& images, 
                           const std::string& directory,
                           const std::string& prefix = "processed_");
cv::Mat loadImage(const std::string& filepath);
void saveImage(const cv::Mat& image, const std::string& filepath);

#endif // IMAGE_PROCESSOR_H
