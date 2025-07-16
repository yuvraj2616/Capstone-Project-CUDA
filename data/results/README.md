# Results Directory

This directory stores the output images and benchmark results from the CUDA Image Processor.

## Directory Structure

```
results/
├── processed_images/          # Output from image processing operations
├── benchmark_data/           # Performance benchmark results
├── batch_output/            # Results from batch processing
└── comparison_images/       # Side-by-side comparisons (CPU vs GPU)
```

## Output File Naming Convention

### Single Image Processing
- **Format**: `{original_name}_{filter}_{parameters}.{extension}`
- **Examples**:
  - `lenna_gaussian_sigma2.0.jpg`
  - `peppers_sobel.png`
  - `mandrill_grayscale.jpg`

### Batch Processing
- **Format**: `{filter}_{index}.{extension}`
- **Examples**:
  - `gaussian_0.png`
  - `sobel_1.png`
  - `grayscale_2.png`

### Benchmark Results
- **Format**: `benchmark_{filter}_{dimensions}_{timestamp}.{extension}`
- **Examples**:
  - `benchmark_gaussian_1920x1080_20240717_143022.csv`
  - `benchmark_summary_20240717_143022.txt`

## Benchmark Data Files

### CSV Format
The benchmark CSV files contain the following columns:
- `Operation`: Filter type (gaussian, sobel, grayscale, hsv)
- `Image_Size_Pixels`: Total number of pixels processed
- `Iterations`: Number of benchmark iterations
- `Avg_Time_ms`: Average processing time in milliseconds
- `Min_Time_ms`: Minimum processing time
- `Max_Time_ms`: Maximum processing time
- `Std_Dev_ms`: Standard deviation of processing times
- `Throughput_MPps`: Throughput in megapixels per second

### Summary Reports
Text-based summary reports include:
- System information (GPU, CPU, memory)
- Test parameters (image size, iterations)
- Performance metrics (speedup, throughput)
- Statistical analysis

## Viewing Results

### Image Viewers
- **Windows**: Windows Photos, Paint, GIMP, Photoshop
- **Cross-platform**: OpenCV (Python), ImageJ, GIMP

### Benchmark Analysis
- **Spreadsheet Software**: Excel, LibreOffice Calc, Google Sheets
- **Data Analysis**: Python (pandas, matplotlib), R, MATLAB
- **Visualization**: Gnuplot, Python matplotlib, Jupyter notebooks

## Performance Analysis Examples

### Loading Benchmark Data in Python
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load benchmark results
df = pd.read_csv('benchmark_summary_20240717_143022.csv')

# Plot performance comparison
plt.figure(figsize=(10, 6))
plt.bar(df['Operation'], df['Speedup'])
plt.xlabel('Operation')
plt.ylabel('GPU Speedup (x)')
plt.title('CUDA vs CPU Performance Comparison')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()
```

### Calculating Performance Metrics
```python
# Throughput analysis
df['Throughput_GPps'] = df['Image_Size_Pixels'] / (df['Avg_Time_ms'] / 1000) / 1e9
print(f"Average GPU throughput: {df['Throughput_GPps'].mean():.2f} GPps")

# Memory bandwidth utilization
image_memory_gb = df['Image_Size_Pixels'] * 3 / 1e9  # 3 bytes per pixel
memory_bandwidth = image_memory_gb / (df['Avg_Time_ms'] / 1000)
print(f"Effective memory bandwidth: {memory_bandwidth.mean():.2f} GB/s")
```

## Quality Assessment

### Image Quality Metrics
You can use various metrics to assess the quality of processed images:

#### Peak Signal-to-Noise Ratio (PSNR)
```python
import cv2
import numpy as np

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Compare original and processed images
original = cv2.imread('original.jpg')
processed = cv2.imread('processed.jpg')
psnr_value = calculate_psnr(original, processed)
print(f"PSNR: {psnr_value:.2f} dB")
```

#### Structural Similarity Index (SSIM)
```python
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

ssim_value = calculate_ssim(original, processed)
print(f"SSIM: {ssim_value:.4f}")
```

## Expected Results

### Performance Expectations
Based on typical GPU configurations:

#### NVIDIA GeForce RTX 3080 (Example)
- **Gaussian Blur**: 50-100x speedup for large images
- **Sobel Edge Detection**: 30-70x speedup
- **Color Conversions**: 80-150x speedup
- **Throughput**: 500-1500 MPps (megapixels per second)

#### Memory Bandwidth Utilization
- **Theoretical Peak**: ~760 GB/s (RTX 3080)
- **Achieved**: 200-400 GB/s (typical for image processing)
- **Efficiency**: 25-50% of theoretical peak

### Image Quality
- **Gaussian Blur**: Identical results to CPU (floating-point precision)
- **Sobel Edge Detection**: Near-identical (minor differences due to parallel execution order)
- **Color Conversions**: Mathematically identical results

## Troubleshooting

### Common Issues
1. **Empty results directory**: Check write permissions and disk space
2. **Corrupted output images**: Verify GPU memory isn't overallocated
3. **Performance inconsistencies**: Ensure consistent GPU load and thermal conditions

### Performance Optimization Tips
1. **Image Size**: Larger images generally show better GPU acceleration
2. **Batch Processing**: More efficient than processing individual images
3. **Memory Management**: Reuse GPU memory allocations when possible
4. **GPU Warmup**: First runs may be slower due to GPU initialization

## Cleanup

To clean up results directory:
```powershell
# Remove all files (Windows PowerShell)
Remove-Item -Path "data\results\*" -Recurse -Force

# Keep directory structure
New-Item -ItemType Directory -Path "data\results\processed_images" -Force
New-Item -ItemType Directory -Path "data\results\benchmark_data" -Force
New-Item -ItemType Directory -Path "data\results\batch_output" -Force
New-Item -ItemType Directory -Path "data\results\comparison_images" -Force
```
