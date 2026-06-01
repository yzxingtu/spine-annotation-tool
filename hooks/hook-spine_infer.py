"""PyInstaller hook for spine_infer.

推理 SDK 在运行时才被 import，需显式收集子模块供 onefile/onedir 打包。
"""

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = collect_submodules("spine_infer")
