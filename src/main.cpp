#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "image_processor.h"
#include "benchmark.h"

namespace fs = std::filesystem;

void printUsage(const std::string& program_name) {
    std::cout << "GPU-Accelerated Image Processing Pipeline\n";
    std::cout << "=========================================\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --input <file>        Input image file path\n";
    std::cout << "  --output <file>       Output image file path\n";
    std::cout << "  --filter <type>       Filter type: gaussian, sobel, grayscale, hsv\n";
    std::cout << "  --sigma <value>       Gaussian filter standard deviation (default: 2.0)\n";
    std::cout << "  --batch               Enable batch processing mode\n";
    std::cout << "  --input_dir <dir>     Input directory for batch processing\n";
    std::cout << "  --output_dir <dir>    Output directory for batch processing\n";
    std::cout << "  --benchmark           Run performance benchmarks\n";
    std::cout << "  --iterations <num>    Number of benchmark iterations (default: 100)\n";
    std::cout << "  --verbose             Enable detailed output\n";
    std::cout << "  --help                Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --input sample.jpg --output blurred.jpg --filter gaussian --sigma 3.0\n";
    std::cout << "  " << program_name << " --batch --input_dir images\\ --output_dir processed\\ --filter sobel\n";
    std::cout << "  " << program_name << " --benchmark --input sample.jpg --iterations 50\n";
}

struct CommandLineArgs {
    std::string input_file;
    std::string output_file;
    std::string filter_type = "gaussian";
    float sigma = 2.0f;
    bool batch_mode = false;
    std::string input_dir;
    std::string output_dir;
    bool benchmark_mode = false;
    int iterations = 100;
    bool verbose = false;
    bool show_help = false;
};

CommandLineArgs parseCommandLine(int argc, char* argv[]) {
    CommandLineArgs args;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            args.show_help = true;
        }
        else if (arg == "--input" && i + 1 < argc) {
            args.input_file = argv[++i];
        }
        else if (arg == "--output" && i + 1 < argc) {
            args.output_file = argv[++i];
        }
        else if (arg == "--filter" && i + 1 < argc) {
            args.filter_type = argv[++i];
        }
        else if (arg == "--sigma" && i + 1 < argc) {
            args.sigma = std::stof(argv[++i]);
        }
        else if (arg == "--batch") {
            args.batch_mode = true;
        }
        else if (arg == "--input_dir" && i + 1 < argc) {
            args.input_dir = argv[++i];
        }
        else if (arg == "--output_dir" && i + 1 < argc) {
            args.output_dir = argv[++i];
        }
        else if (arg == "--benchmark") {
            args.benchmark_mode = true;
        }
        else if (arg == "--iterations" && i + 1 < argc) {
            args.iterations = std::stoi(argv[++i]);
        }
        else if (arg == "--verbose") {
            args.verbose = true;
        }
    }
    
    return args;
}

void runSingleImageProcessing(const CommandLineArgs& args) {
    ImageProcessor processor;
    
    if (args.verbose) {
        std::cout << "Loading image: " << args.input_file << std::endl;
    }
    
    cv::Mat input = loadImage(args.input_file);
    if (input.empty()) {
        std::cerr << "Error: Could not load image " << args.input_file << std::endl;
        return;
    }
    
    if (args.verbose) {
        std::cout << "Image size: " << input.cols << "x" << input.rows 
                  << " channels: " << input.channels() << std::endl;
        std::cout << "Applying filter: " << args.filter_type << std::endl;
    }
    
    cv::Mat result;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (args.filter_type == "gaussian") {
        result = processor.gaussianBlurGPU(input, args.sigma);
    }
    else if (args.filter_type == "sobel") {
        result = processor.sobelEdgeDetectionGPU(input);
    }
    else if (args.filter_type == "grayscale") {
        result = processor.rgbToGrayscaleGPU(input);
    }
    else if (args.filter_type == "hsv") {
        result = processor.rgbToHSVGPU(input);
    }
    else {
        std::cerr << "Error: Unknown filter type: " << args.filter_type << std::endl;
        return;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (args.verbose) {
        std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
    }
    
    std::string output_path = args.output_file;
    if (output_path.empty()) {
        // Generate default output filename
        size_t dot_pos = args.input_file.find_last_of('.');
        std::string base = args.input_file.substr(0, dot_pos);
        std::string ext = args.input_file.substr(dot_pos);
        output_path = base + "_" + args.filter_type + ext;
    }
    
    saveImage(result, output_path);
    
    if (args.verbose) {
        std::cout << "Result saved to: " << output_path << std::endl;
    }
}

void runBatchProcessing(const CommandLineArgs& args) {
    ImageProcessor processor;
    
    if (args.verbose) {
        std::cout << "Loading images from directory: " << args.input_dir << std::endl;
    }
    
    std::vector<cv::Mat> images = loadImagesFromDirectory(args.input_dir);
    
    if (images.empty()) {
        std::cerr << "Error: No images found in directory " << args.input_dir << std::endl;
        return;
    }
    
    if (args.verbose) {
        std::cout << "Found " << images.size() << " images" << std::endl;
        std::cout << "Processing batch with filter: " << args.filter_type << std::endl;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<cv::Mat> results = processor.processBatch(images, args.filter_type, args.sigma);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (args.verbose) {
        std::cout << "Batch processing time: " << duration.count() << " ms" << std::endl;
        std::cout << "Average time per image: " << duration.count() / images.size() << " ms" << std::endl;
    }
    
    std::string output_dir = args.output_dir;
    if (output_dir.empty()) {
        output_dir = "processed_images";
    }
    
    // Create output directory if it doesn't exist
    fs::create_directories(output_dir);
    
    saveImagesToDirectory(results, output_dir, args.filter_type + "_");
    
    if (args.verbose) {
        std::cout << "Results saved to directory: " << output_dir << std::endl;
    }
}

void runBenchmarks(const CommandLineArgs& args) {
    ImageProcessor processor;
    Benchmark benchmark;
    
    std::cout << "Running GPU Performance Benchmarks\n";
    std::cout << "==================================\n\n";
    
    // Print GPU information
    benchmark.printGPUInfo();
    std::cout << std::endl;
    
    cv::Mat input = loadImage(args.input_file);
    if (input.empty()) {
        std::cerr << "Error: Could not load benchmark image " << args.input_file << std::endl;
        return;
    }
    
    std::cout << "Benchmark image: " << input.cols << "x" << input.rows 
              << " (" << input.channels() << " channels)" << std::endl;
    std::cout << "Iterations: " << args.iterations << std::endl << std::endl;
    
    std::vector<BenchmarkResult> results;
    
    // Warm up GPU
    warmupGPU();
    
    // Benchmark each operation
    std::vector<std::string> operations = {"gaussian", "sobel", "grayscale", "hsv"};
    
    for (const auto& op : operations) {
        std::cout << "Benchmarking " << op << " filter..." << std::endl;
        BenchmarkResult result = processor.benchmark(input, op, args.iterations);
        results.push_back(result);
    }
    
    std::cout << std::endl;
    processor.printBenchmarkResults(results);
    
    // Save results to CSV file
    std::string csv_filename = "benchmark_results_" + 
        std::to_string(input.cols) + "x" + std::to_string(input.rows) + ".csv";
    
    // Convert BenchmarkResult to PerformanceMetrics for CSV export
    std::vector<PerformanceMetrics> metrics;
    for (const auto& result : results) {
        PerformanceMetrics metric;
        metric.operation_name = result.operation;
        metric.avg_time_ms = result.gpu_time_ms;
        metric.image_size_pixels = result.image_width * result.image_height;
        metric.throughput_mpixels_per_sec = metric.image_size_pixels / (metric.avg_time_ms * 1000.0);
        metrics.push_back(metric);
    }
    
    benchmark.saveResultsToCSV(metrics, csv_filename);
    std::cout << "Detailed results saved to: " << csv_filename << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        CommandLineArgs args = parseCommandLine(argc, argv);
        
        if (args.show_help || argc == 1) {
            printUsage(argv[0]);
            return 0;
        }
        
        if (args.benchmark_mode) {
            if (args.input_file.empty()) {
                std::cerr << "Error: --input required for benchmark mode" << std::endl;
                return 1;
            }
            runBenchmarks(args);
        }
        else if (args.batch_mode) {
            if (args.input_dir.empty()) {
                std::cerr << "Error: --input_dir required for batch mode" << std::endl;
                return 1;
            }
            runBatchProcessing(args);
        }
        else {
            if (args.input_file.empty()) {
                std::cerr << "Error: --input required for single image processing" << std::endl;
                return 1;
            }
            runSingleImageProcessing(args);
        }
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
