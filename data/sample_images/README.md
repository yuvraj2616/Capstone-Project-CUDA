# Sample Images Directory

This directory contains test images for the CUDA Image Processor project.

## Downloading Test Images

Run the PowerShell script to automatically download standard test images:
```powershell
.\scripts\download_test_data.ps1
```

## Manual Image Sources

If you prefer to manually download images, here are some recommended sources:

### USC SIPI Image Database
- **URL**: https://sipi.usc.edu/database/database.php
- **Description**: Standard test images used in image processing research
- **Recommended Images**:
  - Lenna (5.1.12) - Classic test image
  - Peppers (5.1.10) - Colorful natural image
  - Mandrill (5.1.13) - High detail primate image
  - Goldhill (5.1.09) - Landscape with varied textures

### Creative Commons Images
- **URL**: https://search.creativecommons.org/
- **Description**: Free-to-use images with flexible licensing
- **Recommended Search Terms**: "landscape", "portrait", "texture", "pattern"

### NASA Image Gallery
- **URL**: https://images.nasa.gov/
- **Description**: High-quality space and earth images
- **License**: Public domain

## Image Requirements

- **Supported Formats**: JPEG, PNG, BMP, TIFF
- **Recommended Sizes**: 
  - Small: 512x512 (for quick testing)
  - Medium: 1920x1080 (for standard benchmarks)
  - Large: 4K+ (for performance stress testing)
- **Color Depth**: 8-bit per channel (24-bit RGB)

## Usage Examples

### Single Image Processing
```powershell
# Gaussian blur
.\cuda_image_processor.exe --input data\sample_images\lenna.jpg --output data\results\lenna_blur.jpg --filter gaussian --sigma 3.0

# Edge detection
.\cuda_image_processor.exe --input data\sample_images\peppers.jpg --output data\results\peppers_edges.jpg --filter sobel

# Color conversion
.\cuda_image_processor.exe --input data\sample_images\mandrill.jpg --output data\results\mandrill_gray.jpg --filter grayscale
```

### Batch Processing
```powershell
# Process all images in this directory
.\cuda_image_processor.exe --batch --input_dir data\sample_images --output_dir data\results --filter gaussian --sigma 2.0
```

### Performance Benchmarking
```powershell
# Benchmark with a specific image
.\cuda_image_processor.exe --benchmark --input data\sample_images\lenna.jpg --iterations 100

# Automated benchmark suite
.\scripts\run_benchmarks.ps1 -ImagePath data\sample_images\lenna.jpg -Iterations 50
```

## Creating Synthetic Test Images

If you have Python with OpenCV installed, you can create synthetic test images:

```python
import cv2
import numpy as np

# Create a test pattern
def create_test_pattern(width=1920, height=1080):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Checkerboard pattern
    for i in range(0, height, 50):
        for j in range(0, width, 50):
            if (i//50 + j//50) % 2 == 0:
                image[i:i+50, j:j+50] = [255, 255, 255]
    
    # Add some geometric shapes
    cv2.circle(image, (width//4, height//4), 100, (0, 255, 0), -1)
    cv2.rectangle(image, (width//2, height//4), (width//2 + 200, height//4 + 150), (0, 0, 255), -1)
    
    return image

# Create and save test image
test_image = create_test_pattern()
cv2.imwrite('data/sample_images/synthetic_test.png', test_image)
```

## Performance Considerations

### Image Size Impact on Performance
- **Small images** (< 1MP): CPU might be faster due to GPU setup overhead
- **Medium images** (1-8MP): Sweet spot for GPU acceleration
- **Large images** (> 8MP): Maximum GPU advantage, but watch memory usage

### Memory Requirements
- **RGB Image**: Width × Height × 3 bytes
- **GPU Memory Needed**: ~3x image size (input, output, temporary buffers)
- **Example**: 4K image (3840×2160) needs ~75MB GPU memory

## Troubleshooting

### Common Issues
1. **"Image not found"**: Check file path and supported formats
2. **"GPU memory error"**: Reduce image size or close other GPU applications
3. **"Slow performance"**: Ensure GPU drivers are updated and CUDA is properly installed

### File Format Notes
- **JPEG**: Best for photographs, lossy compression
- **PNG**: Best for graphics with few colors, lossless
- **BMP**: Uncompressed, largest file size
- **TIFF**: High quality, good for scientific images
