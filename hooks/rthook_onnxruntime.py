"""PyInstaller runtime hook: configure DLL paths before application main()."""

import sys

if sys.platform == "win32" and getattr(sys, "frozen", False):
    import os
    from pathlib import Path

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        base = Path(meipass)
        for sub in ("", "onnxruntime/capi", "onnxruntime"):
            directory = base / sub if sub else base
            if directory.is_dir() and hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(directory))
                except OSError:
                    pass
        os.environ["PATH"] = str(base) + os.pathsep + os.environ.get("PATH", "")

        # numpy 须在 onnxruntime 之前加载（否则 frozen 下缺 _multiarray_umath）
        try:
            import numpy  # noqa: F401
        except Exception:
            pass
        try:
            import onnxruntime  # noqa: F401
        except Exception:
            pass
