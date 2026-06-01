"""PyInstaller hook for numpy (required by onnxruntime in frozen builds)."""

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

datas, binaries, hiddenimports = collect_all("numpy")
binaries += collect_dynamic_libs("numpy")

hiddenimports += [
    "numpy.core._multiarray_umath",
    "numpy._core._multiarray_umath",
]
