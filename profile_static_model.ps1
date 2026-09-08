param(
    [string]$ImageDir = "images",
    [string]$Model = "model_trt_static_1108x2232.onnx",
    [ValidateSet("cuda", "tensorrt")]
    [string]$ExecutionProvider = "cuda",
    [double]$DurationSeconds = 120,
    [string]$Output = "two_sender_four_worker_static_120s",
    [string]$NsysPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

if (-not $NsysPath) {
    $NsysPath = "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.1.3\target-windows-x64\nsys.exe"
}

if (-not (Test-Path -LiteralPath $NsysPath -PathType Leaf)) {
    throw "Nsight Systems CLI was not found: $NsysPath"
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $ImageDir) -PathType Container)) {
    throw "Image directory was not found: $ImageDir"
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $Model) -PathType Leaf)) {
    throw "ONNX model was not found: $Model"
}

Push-Location $ProjectRoot
try {
    python analyze_nsys_trace.py profile `
        --image-dir $ImageDir `
        --model $Model `
        --ep $ExecutionProvider `
        --duration-seconds $DurationSeconds `
        --output $Output `
        --nsys-path $NsysPath
} finally {
    Pop-Location
}
