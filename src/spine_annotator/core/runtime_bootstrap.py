"""Windows 下 onnxruntime 加载引导（源码运行与 PyInstaller 打包共用）。"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger("spine_annotator.inference")

_DLL_DIR_HANDLES: List[object] = []
_ORT_LOADED = False


def _meipass_dir() -> Optional[Path]:
    meipass = getattr(sys, "_MEIPASS", None)
    return Path(meipass) if meipass else None


def _iter_dll_search_dirs() -> List[Path]:
    """返回应加入 DLL 搜索路径的目录（去重、仅存在的目录）。"""
    dirs: List[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen or not path.is_dir():
            return
        seen.add(key)
        dirs.append(path)

    meipass = _meipass_dir()
    if meipass:
        add(meipass)
        add(meipass / "onnxruntime" / "capi")
        add(meipass / "onnxruntime")

    import importlib.util

    spec = importlib.util.find_spec("onnxruntime")
    if spec and spec.origin:
        pkg = Path(spec.origin).resolve().parent
        add(pkg)
        add(pkg / "capi")

    return dirs


def configure_windows_dll_search_path() -> None:
    """配置 Windows DLL 搜索路径（可多次调用，幂等）。"""
    if sys.platform != "win32":
        return

    for directory in _iter_dll_search_dirs():
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(str(directory)))
            except OSError:
                LOGGER.debug("add_dll_directory skipped: %s", directory, exc_info=True)

    meipass = _meipass_dir()
    if meipass:
        prefix = str(meipass)
        path_env = os.environ.get("PATH", "")
        if not path_env.startswith(prefix):
            os.environ["PATH"] = prefix + os.pathsep + path_env


def _preload_native_onnx_dlls() -> None:
    """按依赖顺序预加载 onnxruntime 原生 DLL（仅 Windows）。"""
    if sys.platform != "win32":
        return

    import ctypes

    capi_dirs = [d for d in _iter_dll_search_dirs() if (d / "onnxruntime.dll").exists()]
    if not capi_dirs:
        capi_dirs = _iter_dll_search_dirs()

    for capi in capi_dirs:
        for dll_name in ("onnxruntime_providers_shared.dll", "onnxruntime.dll"):
            dll_path = capi / dll_name
            if not dll_path.is_file():
                continue
            try:
                ctypes.WinDLL(str(dll_path))
                LOGGER.debug("Preloaded native DLL: %s", dll_path)
            except OSError as exc:
                LOGGER.warning("Failed to preload %s: %s", dll_path, exc)


def ensure_onnxruntime(*, before_qt: bool = False) -> bool:
    """确保 onnxruntime 已导入；成功返回 True。

    Args:
        before_qt: 为 True 表示在 QApplication 创建前调用（源码模式依赖此顺序）。
    """
    global _ORT_LOADED
    if _ORT_LOADED or "onnxruntime" in sys.modules:
        _ORT_LOADED = True
        return True

    configure_windows_dll_search_path()
    _preload_native_onnx_dlls()

    try:
        # onnxruntime 依赖 numpy C 扩展，frozen 下须先导入 numpy
        import numpy  # noqa: F401
        import onnxruntime as ort  # noqa: F401

        _ORT_LOADED = True
        LOGGER.info(
            "onnxruntime ready: version=%s, frozen=%s, before_qt=%s, file=%s",
            getattr(ort, "__version__", "unknown"),
            getattr(sys, "frozen", False),
            before_qt,
            getattr(ort, "__file__", "unknown"),
        )
        return True
    except Exception:
        LOGGER.error(
            "Failed to load onnxruntime (before_qt=%s, frozen=%s):\n%s",
            before_qt,
            getattr(sys, "frozen", False),
            traceback.format_exc(),
        )
        return False


def preload_onnxruntime_before_qt() -> None:
    """在创建 QApplication 之前预加载 onnxruntime（避免与 Qt 的 DLL 冲突）。"""
    if sys.platform != "win32":
        return

    ok = ensure_onnxruntime(before_qt=True)
    if ok:
        return

    # 打包环境：不在启动阶段终止进程，留待首次推理前再尝试（并写日志）
    if getattr(sys, "frozen", False):
        LOGGER.warning(
            "onnxruntime preload before Qt failed in frozen app; "
            "will retry before AI inference"
        )
        return

    # 源码开发：预加载失败通常意味着 AI 不可用，但不阻断非 AI 功能
    LOGGER.warning(
        "onnxruntime preload before Qt failed; AI inference may not work until resolved"
    )
