# PowerShell script to download sample test images for the CUDA Image Processor
# This script downloads various test images from public sources

Write-Host "CUDA Image Processor - Test Data Setup" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Create directories if they don't exist
$dataDir = "data"
$sampleDir = Join-Path $dataDir "sample_images"
$resultsDir = Join-Path $dataDir "results"

if (!(Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir
    Write-Host "Created directory: $dataDir" -ForegroundColor Yellow
}

if (!(Test-Path $sampleDir)) {
    New-Item -ItemType Directory -Path $sampleDir
    Write-Host "Created directory: $sampleDir" -ForegroundColor Yellow
}

if (!(Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir
    Write-Host "Created directory: $resultsDir" -ForegroundColor Yellow
}

# Function to download a file with progress
function Download-File {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$Description
    )
    
    try {
        Write-Host "Downloading $Description..." -ForegroundColor Cyan
        $webClient = New-Object System.Net.WebClient
        $webClient.DownloadFile($Url, $OutputPath)
        Write-Host "✓ Downloaded: $(Split-Path $OutputPath -Leaf)" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "✗ Failed to download $Description`: $_" -ForegroundColor Red
        return $false
    }
}

# Sample images from USC SIPI Database and other public sources
$testImages = @(
    @{
        Url = "https://sipi.usc.edu/database/download.php?vol=misc&img=5.1.09"
        File = "goldhill.tiff"
        Description = "Goldhill (Standard Test Image)"
    },
    @{
        Url = "https://sipi.usc.edu/database/download.php?vol=misc&img=5.1.10"
        File = "peppers.tiff"
        Description = "Peppers (Standard Test Image)"
    },
    @{
        Url = "https://sipi.usc.edu/database/download.php?vol=misc&img=5.1.12"
        File = "lenna.tiff"
        Description = "Lenna (Standard Test Image)"
    },
    @{
        Url = "https://sipi.usc.edu/database/download.php?vol=misc&img=5.1.13"
        File = "mandrill.tiff"
        Description = "Mandrill (Standard Test Image)"
    }
)

Write-Host "`nDownloading test images..." -ForegroundColor Cyan

$successCount = 0
foreach ($image in $testImages) {
    $outputPath = Join-Path $sampleDir $image.File
    if (Download-File -Url $image.Url -OutputPath $outputPath -Description $image.Description) {
        $successCount++
    }
    Start-Sleep -Milliseconds 500  # Be respectful to the server
}

# Create a simple test pattern if downloads fail
if ($successCount -eq 0) {
    Write-Host "`nCreating synthetic test image..." -ForegroundColor Yellow
    
    $testImageScript = @"
import cv2
import numpy as np

# Create a synthetic test image
width, height = 1920, 1080
image = np.zeros((height, width, 3), dtype=np.uint8)

# Add some patterns
for i in range(0, width, 50):
    cv2.line(image, (i, 0), (i, height), (255, 255, 255), 2)
for i in range(0, height, 50):
    cv2.line(image, (0, i), (width, i), (255, 255, 255), 2)

# Add some geometric shapes
cv2.circle(image, (width//4, height//4), 100, (0, 255, 0), -1)
cv2.rectangle(image, (width//2, height//4), (width//2 + 200, height//4 + 150), (0, 0, 255), -1)
cv2.ellipse(image, (3*width//4, height//4), (120, 80), 45, 0, 360, (255, 0, 0), -1)

# Add noise
noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
image = cv2.add(image, noise)

# Save the test image
cv2.imwrite('$sampleDir/synthetic_test.png', image)
print("Synthetic test image created successfully")
"@

    $pythonScript = Join-Path $env:TEMP "create_test_image.py"
    $testImageScript | Out-File -FilePath $pythonScript -Encoding utf8
    
    try {
        python $pythonScript
        Write-Host "✓ Created synthetic test image" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Could not create synthetic test image. Python with OpenCV required." -ForegroundColor Red
        
        # Create a simple text file as a placeholder
        $placeholderText = @"
To test the CUDA Image Processor, please add image files to this directory.
Supported formats: .jpg, .jpeg, .png, .bmp, .tiff

You can download test images from:
- USC SIPI Database: https://sipi.usc.edu/database/database.php
- Creative Commons: https://search.creativecommons.org/

Example usage:
.\cuda_image_processor.exe --input data\sample_images\your_image.jpg --output data\results\processed.jpg --filter gaussian
"@
        $placeholderPath = Join-Path $sampleDir "README.txt"
        $placeholderText | Out-File -FilePath $placeholderPath -Encoding utf8
        Write-Host "✓ Created placeholder README file" -ForegroundColor Green
    }
    finally {
        if (Test-Path $pythonScript) {
            Remove-Item $pythonScript
        }
    }
}

Write-Host "`n=== Download Summary ===" -ForegroundColor Green
Write-Host "Successfully downloaded: $successCount/$($testImages.Count) images" -ForegroundColor White
Write-Host "Sample images directory: $sampleDir" -ForegroundColor White
Write-Host "Results directory: $resultsDir" -ForegroundColor White

Write-Host "`n=== Next Steps ===" -ForegroundColor Green
Write-Host "1. Build the project using CMake or Visual Studio" -ForegroundColor White
Write-Host "2. Run basic test: .\cuda_image_processor.exe --input data\sample_images\[image] --filter gaussian" -ForegroundColor White
Write-Host "3. Run benchmarks: .\cuda_image_processor.exe --benchmark --input data\sample_images\[image]" -ForegroundColor White

Write-Host "`nSetup completed!" -ForegroundColor Green
