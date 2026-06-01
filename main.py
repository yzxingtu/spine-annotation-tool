"""Spine Annotator - OBB annotation tool for spine X-ray images."""

import logging
import os
import sys
import traceback
from pathlib import Path

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

_DLL_DIR_HANDLES = []


def _setup_logging() -> None:
    """Configure console logging for startup and inference diagnostics."""
    level_name = os.getenv("SPINE_ANNOTATOR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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


def _preload_onnxruntime_for_windows() -> None:
    """Preload onnxruntime before QApplication to avoid DLL init conflicts on Windows."""
    logger = logging.getLogger("spine_annotator.inference")
    if sys.platform != "win32":
        return

    try:
        import importlib.util

        spec = importlib.util.find_spec("onnxruntime")
        if spec is None or spec.origin is None:
            logger.warning("onnxruntime not found during preload step")
            return

        ort_pkg_dir = Path(spec.origin).resolve().parent
        ort_capi_dir = ort_pkg_dir / "capi"
        if hasattr(os, "add_dll_directory") and ort_capi_dir.exists():
            # Keep handle alive for whole process lifetime.
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(ort_capi_dir)))
            logger.info("Added ONNXRuntime DLL directory: %s", ort_capi_dir)

        import onnxruntime as ort

        logger.info(
            "Preloaded onnxruntime before Qt init: version=%s, file=%s",
            getattr(ort, "__version__", "unknown"),
            getattr(ort, "__file__", "unknown"),
        )
    except Exception:
        logger.error(
            "Failed to preload onnxruntime before Qt init:\n%s",
            traceback.format_exc(),
        )


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
