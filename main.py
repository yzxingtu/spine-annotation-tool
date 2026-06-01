"""Spine Annotator - OBB annotation tool for spine X-ray images."""

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# Add src to path for development
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 供 PyInstaller 静态分析依赖（实际推理在 spine_annotator.core.inference 中延迟 import）
if False:  # pragma: no cover
    import onnxruntime  # noqa: F401
    import spine_infer  # noqa: F401

_DLL_DIR_HANDLES = []


def _setup_logging() -> None:
    """Configure console logging for startup and inference diagnostics."""
    level_name = os.getenv("SPINE_ANNOTATOR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if getattr(sys, "frozen", False):
        log_dir = Path.home() / ".cache" / "spine-annotator" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("spine_annotator").info(
        "Logging initialized (level=%s, python=%s, platform=%s)",
        logging.getLevelName(level),
        sys.version.split()[0],
        sys.platform,
    )
    logging.getLogger("spine_annotator").info(
        "Runtime context: exe=%s, cwd=%s",
        sys.executable,
        os.getcwd(),
    )
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    logging.getLogger("spine_annotator").info("PATH head entries: %s", path_entries[:8])


def _resolve_onnxruntime_capi_dir() -> Optional[Path]:
    """Return onnxruntime/capi directory for dev and PyInstaller frozen layouts."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        capi = Path(meipass) / "onnxruntime" / "capi"
        if capi.is_dir():
            return capi

    import importlib.util

    spec = importlib.util.find_spec("onnxruntime")
    if spec is None or not spec.origin:
        return None
    capi = Path(spec.origin).resolve().parent / "capi"
    return capi if capi.is_dir() else None


def _preload_onnxruntime_for_windows() -> None:
    """Preload onnxruntime before QApplication to avoid DLL init conflicts on Windows."""
    logger = logging.getLogger("spine_annotator.inference")
    if sys.platform != "win32":
        return

    try:
        ort_capi_dir = _resolve_onnxruntime_capi_dir()
        if ort_capi_dir is None:
            logger.warning("onnxruntime capi directory not found during preload step")
            return

        if hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(ort_capi_dir)))
            logger.info("Added ONNXRuntime DLL directory: %s", ort_capi_dir)

        import onnxruntime as ort

        logger.info(
            "Preloaded onnxruntime before Qt init: version=%s, frozen=%s, file=%s",
            getattr(ort, "__version__", "unknown"),
            getattr(sys, "frozen", False),
            getattr(ort, "__file__", "unknown"),
        )
    except Exception:
        logger.error(
            "Failed to preload onnxruntime before Qt init:\n%s",
            traceback.format_exc(),
        )
        if getattr(sys, "frozen", False):
            raise


def _install_excepthook() -> None:
    """Log uncaught exceptions to console for faster debugging."""
    logger = logging.getLogger("spine_annotator")

    def _hook(exc_type, exc_value, exc_tb):
        logger.error(
            "Uncaught exception:\n%s",
            "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook


def main():
    """Main entry point."""
    _setup_logging()
    _install_excepthook()
    _preload_onnxruntime_for_windows()
    # Import Qt/UI after onnxruntime preload to avoid DLL init conflicts on Windows.
    from PyQt5.QtWidgets import QApplication
    from spine_annotator.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Spine Annotator")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
