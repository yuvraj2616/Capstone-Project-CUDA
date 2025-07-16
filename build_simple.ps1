# Simple CUDA Image Processor Build Script
param(
    [string]$BuildType = "Release"
)

Write-Host "Building CUDA Image Processor..." -ForegroundColor Green

# Check for required tools
Write-Host "Checking prerequisites..." -ForegroundColor Cyan

# Check CMake
try {
    $cmake = cmake --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ CMake found" -ForegroundColor Green
    } else {
        throw "CMake not found"
    }
} catch {
    Write-Host "✗ CMake not found. Please install CMake." -ForegroundColor Red
    exit 1
}

# Check NVCC
try {
    $nvcc = nvcc --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ CUDA Toolkit found" -ForegroundColor Green
    } else {
        throw "NVCC not found"
    }
} catch {
    Write-Host "✗ CUDA Toolkit not found. Please install CUDA Toolkit." -ForegroundColor Red
    exit 1
}

# Create build directory
$buildDir = "build"
if (!(Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
    Write-Host "Created build directory" -ForegroundColor Yellow
}

Set-Location $buildDir

# Configure with CMake (simplified)
Write-Host "Configuring project..." -ForegroundColor Cyan
try {
    cmake .. -A x64 -DCMAKE_BUILD_TYPE=$BuildType
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-Host "✓ Configuration successful" -ForegroundColor Green
} catch {
    Write-Host "✗ Configuration failed: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Build
Write-Host "Building project..." -ForegroundColor Cyan
try {
    cmake --build . --config $BuildType
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    Write-Host "✓ Build successful" -ForegroundColor Green
} catch {
    Write-Host "✗ Build failed: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

Set-Location ..

# Check for executable
$exePath = "build\$BuildType\cuda_image_processor.exe"
if (Test-Path $exePath) {
    Write-Host "✓ Executable created: $exePath" -ForegroundColor Green
} else {
    Write-Host "✗ Executable not found" -ForegroundColor Red
}

Write-Host "Build completed!" -ForegroundColor Green
