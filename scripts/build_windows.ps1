# Local Windows build (aligned with CI: numpy<2 + onnxruntime 1.17.3)
# Usage: .\scripts\build_windows.ps1
# Close any running "X光脊柱标注工具.exe" before building (dist folder lock).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
$PyInstaller = Join-Path $Root ".venv\Scripts\pyinstaller.exe"
$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
$Trusted = @("--trusted-host", "pypi.tuna.tsinghua.edu.cn")

if (-not (Test-Path $Python)) {
    Write-Error "Missing .venv. Create venv and install dependencies first."
}

Write-Host "[INFO] Installing build stack: numpy<2, onnxruntime==1.17.3 ..."
& $Python -m pip install pyinstaller "numpy>=1.21.0,<2.0" "onnxruntime==1.17.3" "opencv-python-headless>=4.5.0,<4.10" -i $Mirror @Trusted -q

& $Python -c "import numpy; import onnxruntime; print('numpy', numpy.__version__, 'ort', onnxruntime.__version__)"

Write-Host "[INFO] PyInstaller build starting..."
& $PyInstaller --noconfirm --clean `
    --name "X光脊柱标注工具" `
    --windowed `
    --paths src `
    --collect-submodules spine_annotator `
    --collect-submodules spine_infer `
    --collect-all onnxruntime `
    --collect-all numpy `
    --collect-all spine_infer `
    --hidden-import spine_infer `
    --hidden-import spine_infer.backends.onnx_backend `
    --hidden-import numpy.core._multiarray_umath `
    --hidden-import numpy._core._multiarray_umath `
    --additional-hooks-dir hooks `
    --runtime-hook hooks/rthook_onnxruntime.py `
    main.py

$Dist = Join-Path $Root "dist\X光脊柱标注工具"
$Exe = Join-Path $Dist "X光脊柱标注工具.exe"
if (Test-Path $Exe) {
    Write-Host "[OK] Build done: $Exe"
    Write-Host "[OK] Test the whole folder: dist\X光脊柱标注工具\"
} else {
    Write-Error "exe not found under dist. Check PyInstaller output."
}
