cmake_minimum_required(VERSION 3.18)
project(CUDAImageProcessor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# Check CUDA compute capability
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 50 60 70 75 80 86)
endif()

# Compiler-specific options
if(MSVC)
    # Visual Studio specific settings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
else()
    # GCC/Clang settings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Source files
set(CUDA_SOURCES
    src/cuda_kernels.cu
)

set(CPP_SOURCES
    src/main.cpp
    src/image_processor.cpp
    src/benchmark.cpp
)

# Create the executable
add_executable(cuda_image_processor
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
)

# Set properties for CUDA compilation
set_property(TARGET cuda_image_processor PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link libraries
target_link_libraries(cuda_image_processor
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
)

# Platform-specific linking
if(WIN32)
    target_link_libraries(cuda_image_processor
        cudart
        curand
    )
else()
    target_link_libraries(cuda_image_processor
        ${CUDA_cudart_LIBRARY}
        ${CUDA_curand_LIBRARY}
    )
endif()

# Compiler definitions
target_compile_definitions(cuda_image_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_OPERATORS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF_CONVERSIONS__>
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDA_NO_HALF2_OPERATORS__>
)

# Installation
install(TARGETS cuda_image_processor
    RUNTIME DESTINATION bin
)

# Create data directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data/sample_images)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/data/results)

# Copy sample data (if exists)
if(EXISTS ${CMAKE_SOURCE_DIR}/data)
    file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR}/data/)
endif()

# Print configuration summary
message(STATUS "=====================================")
message(STATUS "CUDA Image Processor Configuration")
message(STATUS "=====================================")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")
message(STATUS "=====================================")

# Optional: Build tests (if test files exist)
option(BUILD_TESTS "Build test programs" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Optional: Build documentation
option(BUILD_DOCS "Build documentation" OFF)
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        configure_file(${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in 
                       ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
        add_custom_target(docs
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen" VERBATIM
        )
    endif()
endif()
