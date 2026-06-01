"""Spine Annotator - OBB annotation tool for spine X-ray images."""

import logging
import os
import sys
import traceback
from pathlib import Path

# Add src to path for development
if not getattr(sys, "frozen", False):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 供 PyInstaller 静态分析依赖（实际推理在 spine_annotator.core.inference 中延迟 import）
if False:  # pragma: no cover
    import numpy  # noqa: F401
    import onnxruntime  # noqa: F401
    import spine_infer  # noqa: F401


def _setup_logging() -> None:
    """Configure console logging for startup and inference diagnostics."""
    level_name = os.getenv("SPINE_ANNOTATOR_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handlers = [logging.StreamHandler()]
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
        "Logging initialized (level=%s, python=%s, platform=%s, frozen=%s)",
        logging.getLevelName(level),
        sys.version.split()[0],
        sys.platform,
        getattr(sys, "frozen", False),
    )
    logging.getLogger("spine_annotator").info(
        "Runtime context: exe=%s, cwd=%s",
        sys.executable,
        os.getcwd(),
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

    from spine_annotator.core.runtime_bootstrap import preload_onnxruntime_before_qt

    preload_onnxruntime_before_qt()

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
