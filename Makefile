# CUDA Image Processor Makefile
# Alternative build system for environments where CMake is not preferred

# Compiler settings
NVCC = nvcc
CXX = g++

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

# OpenCV settings (adjust paths as needed)
OPENCV_CFLAGS = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`

# CUDA settings
CUDA_ARCH = -arch=sm_50 -arch=sm_60 -arch=sm_70 -arch=sm_75 -arch=sm_80 -arch=sm_86
CUDA_FLAGS = -O3 --use_fast_math $(CUDA_ARCH) -I$(INCDIR)
CXX_FLAGS = -O3 -std=c++17 -I$(INCDIR) $(OPENCV_CFLAGS)

# Source files
CUDA_SOURCES = $(SRCDIR)/cuda_kernels.cu
CPP_SOURCES = $(SRCDIR)/main.cpp $(SRCDIR)/image_processor.cpp $(SRCDIR)/benchmark.cpp

# Object files
CUDA_OBJECTS = $(OBJDIR)/cuda_kernels.o
CPP_OBJECTS = $(OBJDIR)/main.o $(OBJDIR)/image_processor.o $(OBJDIR)/benchmark.o

# Target executable
TARGET = $(BINDIR)/cuda_image_processor

# Default target
all: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(OBJDIR) $(BINDIR) data/sample_images data/results

# Link the final executable
$(TARGET): $(CUDA_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $^ $(OPENCV_LIBS) -lcudart -lcurand

# Compile CUDA source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@

# Compile C++ source files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Install (copy to system directory)
install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

# Uninstall
uninstall:
	rm -f /usr/local/bin/cuda_image_processor

# Run basic tests
test: $(TARGET)
	@echo "Running basic functionality tests..."
	@if [ -f data/sample_images/test.jpg ]; then \
		$(TARGET) --input data/sample_images/test.jpg --output data/results/test_gaussian.jpg --filter gaussian; \
		$(TARGET) --input data/sample_images/test.jpg --output data/results/test_sobel.jpg --filter sobel; \
		$(TARGET) --input data/sample_images/test.jpg --output data/results/test_grayscale.jpg --filter grayscale; \
		echo "Tests completed. Check data/results/ for output images."; \
	else \
		echo "No test image found. Please add a test image to data/sample_images/test.jpg"; \
	fi

# Benchmark with default image
benchmark: $(TARGET)
	@echo "Running performance benchmarks..."
	@if [ -f data/sample_images/test.jpg ]; then \
		$(TARGET) --benchmark --input data/sample_images/test.jpg --iterations 50; \
	else \
		echo "No test image found. Please add a test image to data/sample_images/test.jpg"; \
	fi

# Create sample data directory structure
setup:
	@echo "Setting up project structure..."
	@mkdir -p data/sample_images data/results scripts
	@echo "Project structure created."
	@echo "Please add sample images to data/sample_images/"

# Debug build (with debug symbols)
debug: CUDA_FLAGS = -g -G -I$(INCDIR)
debug: CXX_FLAGS = -g -std=c++17 -I$(INCDIR) $(OPENCV_CFLAGS)
debug: directories $(TARGET)

# Profile build (for profiling tools like nvprof)
profile: CUDA_FLAGS = -O3 --use_fast_math $(CUDA_ARCH) -I$(INCDIR) -lineinfo
profile: CXX_FLAGS = -O3 -std=c++17 -I$(INCDIR) $(OPENCV_CFLAGS)
profile: directories $(TARGET)

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@which nvcc > /dev/null || (echo "NVCC not found. Please install CUDA Toolkit." && exit 1)
	@which $(CXX) > /dev/null || (echo "$(CXX) not found. Please install a C++ compiler." && exit 1)
	@pkg-config --exists opencv4 || (echo "OpenCV 4.x not found. Please install OpenCV development packages." && exit 1)
	@echo "All dependencies found."

# Show help
help:
	@echo "Available targets:"
	@echo "  all       - Build the project (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to system directory"
	@echo "  uninstall - Remove from system directory"
	@echo "  test      - Run basic functionality tests"
	@echo "  benchmark - Run performance benchmarks"
	@echo "  setup     - Create project directory structure"
	@echo "  debug     - Build with debug symbols"
	@echo "  profile   - Build optimized for profiling"
	@echo "  check-deps- Check for required dependencies"
	@echo "  help      - Show this help message"

# Phony targets
.PHONY: all clean install uninstall test benchmark setup debug profile check-deps help directories

# Dependencies (simplified)
$(OBJDIR)/main.o: $(INCDIR)/image_processor.h $(INCDIR)/benchmark.h
$(OBJDIR)/image_processor.o: $(INCDIR)/image_processor.h $(INCDIR)/cuda_kernels.cuh $(INCDIR)/benchmark.h
$(OBJDIR)/benchmark.o: $(INCDIR)/benchmark.h
$(OBJDIR)/cuda_kernels.o: $(INCDIR)/cuda_kernels.cuh
