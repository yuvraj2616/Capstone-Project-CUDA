# CUDA Image Processor - Windows Build Script
# This script automates the build process using CMake and Visual Studio

param(
    [string]$BuildType = "Release",
    [string]$Generator = "Visual Studio 17 2022",
    [string]$VcpkgPath = "",
    [switch]$Clean,
    [switch]$Install,
    [switch]$Test,
    [switch]$Help
)

if ($Help) {
    Write-Host "CUDA Image Processor Build Script" -ForegroundColor Green
    Write-Host "=================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage: .\build.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -BuildType <type>     Build configuration (Release, Debug, RelWithDebInfo)"
    Write-Host "  -Generator <gen>      CMake generator (default: Visual Studio 17 2022)"
    Write-Host "  -VcpkgPath <path>     Path to vcpkg installation"
    Write-Host "  -Clean               Clean build directory before building"
    Write-Host "  -Install             Install after successful build"
    Write-Host "  -Test                Run tests after build"
    Write-Host "  -Help                Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\build.ps1                                    # Standard release build"
    Write-Host "  .\build.ps1 -BuildType Debug                   # Debug build"
    Write-Host "  .\build.ps1 -Clean -Install                    # Clean build and install"
    Write-Host "  .\build.ps1 -VcpkgPath C:\vcpkg               # Use specific vcpkg path"
    exit 0
}

Write-Host "CUDA Image Processor - Build Script" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Cyan

# Check for CMake
try {
    $cmakeVersion = cmake --version | Select-Object -First 1
    Write-Host "✓ CMake found: $cmakeVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ CMake not found. Please install CMake 3.18 or later." -ForegroundColor Red
    Write-Host "  Download from: https://cmake.org/download/" -ForegroundColor Yellow
    exit 1
}

# Check for CUDA
try {
    $nvccVersion = nvcc --version | Select-String "release"
    Write-Host "✓ CUDA found: $nvccVersion" -ForegroundColor Green
}
catch {
    Write-Host "✗ CUDA Toolkit not found. Please install CUDA Toolkit 11.0 or later." -ForegroundColor Red
    Write-Host "  Download from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    exit 1
}

# Check for Visual Studio
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (Test-Path $vsWhere) {
    $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsInstall) {
        Write-Host "✓ Visual Studio found: $vsInstall" -ForegroundColor Green
    } else {
        Write-Host "✗ Visual Studio with C++ tools not found." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "⚠ Could not verify Visual Studio installation" -ForegroundColor Yellow
}

# Auto-detect vcpkg if not provided
if ($VcpkgPath -eq "") {
    $possiblePaths = @(
        "C:\vcpkg",
        "C:\tools\vcpkg",
        "$env:VCPKG_ROOT",
        ".\vcpkg"
    )
    
    foreach ($path in $possiblePaths) {
        if ($path -and (Test-Path "$path\scripts\buildsystems\vcpkg.cmake")) {
            $VcpkgPath = $path
            break
        }
    }
}

if ($VcpkgPath -and (Test-Path "$VcpkgPath\scripts\buildsystems\vcpkg.cmake")) {
    Write-Host "✓ vcpkg found: $VcpkgPath" -ForegroundColor Green
    $vcpkgToolchain = "$VcpkgPath\scripts\buildsystems\vcpkg.cmake"
} else {
    Write-Host "⚠ vcpkg not found. OpenCV must be installed manually." -ForegroundColor Yellow
    $vcpkgToolchain = ""
}

# Set up build directory
$buildDir = "build"
if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "`nCleaning build directory..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $buildDir
}

if (!(Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Set-Location $buildDir

# Configure with CMake
Write-Host "`nConfiguring project..." -ForegroundColor Cyan

$cmakeArgs = @(
    "..",
    "-G", $Generator,
    "-A", "x64",
    "-DCMAKE_BUILD_TYPE=$BuildType"
)

if ($vcpkgToolchain) {
    $cmakeArgs += "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain"
}

try {
    & cmake $cmakeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed"
    }
    Write-Host "✓ Configuration completed successfully" -ForegroundColor Green
}
catch {
    Write-Host "✗ Configuration failed: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Build the project
Write-Host "`nBuilding project..." -ForegroundColor Cyan

try {
    & cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed"
    }
    Write-Host "✓ Build completed successfully" -ForegroundColor Green
}
catch {
    Write-Host "✗ Build failed: $_" -ForegroundColor Red
    Set-Location ..
    exit 1
}

# Install if requested
if ($Install) {
    Write-Host "`nInstalling..." -ForegroundColor Cyan
    try {
        & cmake --install . --config $BuildType
        if ($LASTEXITCODE -ne 0) {
            throw "Installation failed"
        }
        Write-Host "✓ Installation completed successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Installation failed: $_" -ForegroundColor Red
    }
}

Set-Location ..

# Verify build output
$exePath = "build\$BuildType\cuda_image_processor.exe"
if (Test-Path $exePath) {
    Write-Host "`n✓ Executable created: $exePath" -ForegroundColor Green
    
    # Get file size
    $fileSize = (Get-Item $exePath).Length
    $fileSizeMB = [math]::Round($fileSize / 1MB, 2)
    Write-Host "  File size: $fileSizeMB MB" -ForegroundColor White
} else {
    Write-Host "`n✗ Executable not found at expected location" -ForegroundColor Red
    Write-Host "  Expected: $exePath" -ForegroundColor Yellow
}

# Run basic test if requested
if ($Test -and (Test-Path $exePath)) {
    Write-Host "`nRunning basic functionality test..." -ForegroundColor Cyan
    
    # Create a simple test image
    $testScript = @"
import cv2
import numpy as np
img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
cv2.imwrite('test_image.png', img)
"@
    
    try {
        python -c $testScript
        if (Test-Path "test_image.png") {
            & $exePath --input test_image.png --output test_output.png --filter gaussian --sigma 1.0
            if ($LASTEXITCODE -eq 0 -and (Test-Path "test_output.png")) {
                Write-Host "✓ Basic functionality test passed" -ForegroundColor Green
                Remove-Item "test_image.png", "test_output.png" -ErrorAction SilentlyContinue
            } else {
                Write-Host "✗ Basic functionality test failed" -ForegroundColor Red
            }
        }
    }
    catch {
        Write-Host "⚠ Could not run basic test (Python/OpenCV required)" -ForegroundColor Yellow
    }
}

Write-Host "`n=== Build Summary ===" -ForegroundColor Green
Write-Host "Build Type: $BuildType" -ForegroundColor White
Write-Host "Generator: $Generator" -ForegroundColor White
Write-Host "Executable: $exePath" -ForegroundColor White

if ($vcpkgToolchain) {
    Write-Host "vcpkg: $VcpkgPath" -ForegroundColor White
}

Write-Host "`n=== Next Steps ===" -ForegroundColor Green
Write-Host "1. Download test data: .\scripts\download_test_data.ps1" -ForegroundColor White
Write-Host "2. Run basic test: $exePath --input data\sample_images\[image] --filter gaussian" -ForegroundColor White
Write-Host "3. Run benchmarks: .\scripts\run_benchmarks.ps1" -ForegroundColor White

Write-Host "`nBuild completed successfully!" -ForegroundColor Green
