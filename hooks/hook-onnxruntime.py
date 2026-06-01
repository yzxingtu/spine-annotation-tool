"""PyInstaller hook for onnxruntime.

显式收集 onnxruntime/capi/ 下的所有二进制文件（.pyd, .dll），
解决 Windows 上 DLL 加载失败的问题。
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

binaries = collect_dynamic_libs("onnxruntime")
datas = collect_data_files("onnxruntime")
