# PowerShell script to run comprehensive benchmarks for the CUDA Image Processor
# This script automates the benchmarking process and generates detailed reports

param(
    [string]$ImagePath = "",
    [int]$Iterations = 100,
    [string]$OutputDir = "benchmark_results",
    [switch]$Verbose
)

Write-Host "CUDA Image Processor - Automated Benchmarks" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Check if the executable exists
$exePath = "cuda_image_processor.exe"
if (!(Test-Path $exePath)) {
    $exePath = "bin\cuda_image_processor.exe"
    if (!(Test-Path $exePath)) {
        $exePath = "build\Release\cuda_image_processor.exe"
        if (!(Test-Path $exePath)) {
            Write-Host "Error: cuda_image_processor.exe not found. Please build the project first." -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host "Using executable: $exePath" -ForegroundColor Yellow

# Create output directory
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "Created output directory: $OutputDir" -ForegroundColor Yellow
}

# Find test images if not specified
if ($ImagePath -eq "") {
    $sampleDir = "data\sample_images"
    if (Test-Path $sampleDir) {
        $images = Get-ChildItem -Path $sampleDir -Include *.jpg,*.jpeg,*.png,*.bmp,*.tiff -Recurse
        if ($images.Count -gt 0) {
            $ImagePath = $images[0].FullName
            Write-Host "Using test image: $ImagePath" -ForegroundColor Yellow
        }
    }
    
    if ($ImagePath -eq "") {
        Write-Host "Error: No test image specified and none found in data\sample_images\" -ForegroundColor Red
        Write-Host "Please specify an image path with -ImagePath parameter or run download_test_data.ps1 first" -ForegroundColor Red
        exit 1
    }
}

# Check if image exists
if (!(Test-Path $ImagePath)) {
    Write-Host "Error: Image file not found: $ImagePath" -ForegroundColor Red
    exit 1
}

# Get image information
try {
    Add-Type -AssemblyName System.Drawing
    $image = [System.Drawing.Image]::FromFile((Resolve-Path $ImagePath))
    $imageWidth = $image.Width
    $imageHeight = $image.Height
    $image.Dispose()
    Write-Host "Image dimensions: ${imageWidth}x${imageHeight}" -ForegroundColor Cyan
}
catch {
    Write-Host "Warning: Could not read image dimensions" -ForegroundColor Yellow
    $imageWidth = "Unknown"
    $imageHeight = "Unknown"
}

# Function to run a benchmark and capture output
function Run-Benchmark {
    param(
        [string]$Filter,
        [string]$Description
    )
    
    Write-Host "`nRunning $Description benchmark..." -ForegroundColor Cyan
    
    $outputFile = Join-Path $OutputDir "benchmark_${Filter}_${imageWidth}x${imageHeight}.txt"
    $csvFile = Join-Path $OutputDir "benchmark_${Filter}_${imageWidth}x${imageHeight}.csv"
    
    $startTime = Get-Date
    
    try {
        if ($Verbose) {
            $arguments = @("--benchmark", "--input", $ImagePath, "--iterations", $Iterations, "--verbose")
        } else {
            $arguments = @("--benchmark", "--input", $ImagePath, "--iterations", $Iterations)
        }
        
        $result = & $exePath $arguments 2>&1 | Tee-Object -FilePath $outputFile
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        Write-Host "✓ $Description completed in $([math]::Round($duration, 2)) seconds" -ForegroundColor Green
        
        # Extract timing information from output
        $gpuTime = $null
        $cpuTime = $null
        $speedup = $null
        
        foreach ($line in $result) {
            if ($line -match "GPU Time.*?(\d+\.?\d*)\s*ms") {
                $gpuTime = $matches[1]
            }
            if ($line -match "CPU Time.*?(\d+\.?\d*)\s*ms") {
                $cpuTime = $matches[1]
            }
            if ($line -match "Speedup.*?(\d+\.?\d*)x") {
                $speedup = $matches[1]
            }
        }
        
        return @{
            Filter = $Filter
            Description = $Description
            Duration = $duration
            GPUTime = $gpuTime
            CPUTime = $cpuTime
            Speedup = $speedup
            OutputFile = $outputFile
        }
    }
    catch {
        Write-Host "✗ $Description failed: $_" -ForegroundColor Red
        return $null
    }
}

# Run benchmarks for all filters
$filters = @(
    @{ Name = "gaussian"; Description = "Gaussian Blur" },
    @{ Name = "sobel"; Description = "Sobel Edge Detection" },
    @{ Name = "grayscale"; Description = "RGB to Grayscale" },
    @{ Name = "hsv"; Description = "RGB to HSV" }
)

Write-Host "`nStarting comprehensive benchmark suite..." -ForegroundColor Green
Write-Host "Image: $ImagePath" -ForegroundColor White
Write-Host "Iterations per test: $Iterations" -ForegroundColor White
Write-Host "Output directory: $OutputDir" -ForegroundColor White

$benchmarkResults = @()

foreach ($filter in $filters) {
    $result = Run-Benchmark -Filter $filter.Name -Description $filter.Description
    if ($result) {
        $benchmarkResults += $result
    }
}

# Generate summary report
Write-Host "`n=== Benchmark Summary ===" -ForegroundColor Green

$summaryFile = Join-Path $OutputDir "benchmark_summary_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
$csvSummaryFile = Join-Path $OutputDir "benchmark_summary_$(Get-Date -Format 'yyyyMMdd_HHmmss').csv"

$summaryContent = @"
CUDA Image Processor Benchmark Summary
=====================================
Date: $(Get-Date)
Image: $ImagePath
Dimensions: ${imageWidth}x${imageHeight}
Iterations: $Iterations

Results:
--------
"@

$csvContent = "Filter,Description,GPU_Time_ms,CPU_Time_ms,Speedup,Total_Duration_sec`n"

foreach ($result in $benchmarkResults) {
    $line = "{0,-15} | GPU: {1,8} ms | CPU: {2,8} ms | Speedup: {3,6}x" -f 
            $result.Description, 
            ($result.GPUTime ?? "N/A"), 
            ($result.CPUTime ?? "N/A"), 
            ($result.Speedup ?? "N/A")
    
    $summaryContent += "`n$line"
    Write-Host $line -ForegroundColor White
    
    $csvLine = "{0},{1},{2},{3},{4},{5}" -f 
               $result.Filter, 
               $result.Description, 
               ($result.GPUTime ?? ""), 
               ($result.CPUTime ?? ""), 
               ($result.Speedup ?? ""), 
               $result.Duration
    $csvContent += "$csvLine`n"
}

# Save summary
$summaryContent | Out-File -FilePath $summaryFile -Encoding utf8
$csvContent | Out-File -FilePath $csvSummaryFile -Encoding utf8

Write-Host "`n=== Performance Analysis ===" -ForegroundColor Green

$validResults = $benchmarkResults | Where-Object { $_.Speedup -ne $null -and $_.Speedup -ne "N/A" }
if ($validResults.Count -gt 0) {
    $avgSpeedup = ($validResults | ForEach-Object { [double]$_.Speedup } | Measure-Object -Average).Average
    $maxSpeedup = ($validResults | ForEach-Object { [double]$_.Speedup } | Measure-Object -Maximum).Maximum
    $minSpeedup = ($validResults | ForEach-Object { [double]$_.Speedup } | Measure-Object -Minimum).Minimum
    
    Write-Host "Average GPU Speedup: $([math]::Round($avgSpeedup, 2))x" -ForegroundColor Cyan
    Write-Host "Maximum GPU Speedup: $([math]::Round($maxSpeedup, 2))x" -ForegroundColor Cyan
    Write-Host "Minimum GPU Speedup: $([math]::Round($minSpeedup, 2))x" -ForegroundColor Cyan
}

Write-Host "`n=== Output Files ===" -ForegroundColor Green
Write-Host "Summary Report: $summaryFile" -ForegroundColor White
Write-Host "CSV Data: $csvSummaryFile" -ForegroundColor White
Write-Host "Individual Results: $OutputDir\benchmark_*.txt" -ForegroundColor White

# System information
Write-Host "`n=== System Information ===" -ForegroundColor Green
try {
    $gpu = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" } | Select-Object -First 1
    if ($gpu) {
        Write-Host "GPU: $($gpu.Name)" -ForegroundColor White
        Write-Host "GPU Memory: $([math]::Round($gpu.AdapterRAM / 1GB, 2)) GB" -ForegroundColor White
    }
    
    $cpu = Get-WmiObject -Class Win32_Processor | Select-Object -First 1
    Write-Host "CPU: $($cpu.Name)" -ForegroundColor White
    Write-Host "CPU Cores: $($cpu.NumberOfCores)" -ForegroundColor White
    Write-Host "RAM: $([math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)) GB" -ForegroundColor White
}
catch {
    Write-Host "Could not retrieve system information" -ForegroundColor Yellow
}

Write-Host "`nBenchmark suite completed!" -ForegroundColor Green
