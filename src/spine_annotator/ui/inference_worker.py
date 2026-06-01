"""异步推理 Worker：在后台线程执行模型下载 + 推理，避免阻塞 UI。"""

from __future__ import annotations

import logging
import os
import time
import traceback
from pathlib import Path
from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from ..core.inference import ModelManager, SpineInferenceBridge
from ..core.models import OBBAnnotation

LOGGER = logging.getLogger("spine_annotator.inference")


class InferenceWorker(QThread):
    """后台推理线程。

    Signals:
        finished: 推理成功，携带 OBBAnnotation 列表
        error:    推理失败，携带错误信息
        progress: 进度更新（模型下载时）
    """

    finished = pyqtSignal(list)     # List[OBBAnnotation]
    error = pyqtSignal(str)         # 错误信息
    progress = pyqtSignal(str)      # 进度文字

    def __init__(
        self,
        image_path: str,
        model_manager: ModelManager,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._model_manager = model_manager

    def run(self) -> None:  # noqa: D401
        try:
            start_ts = time.perf_counter()
            image_file = Path(self._image_path)
            LOGGER.info(
                "Inference worker started: image=%s, thread=%s",
                self._image_path,
                int(self.currentThreadId()),
            )
            LOGGER.info(
                "Image file check: exists=%s, size=%s bytes",
                image_file.exists(),
                image_file.stat().st_size if image_file.exists() else "N/A",
            )
            if not image_file.exists():
                raise FileNotFoundError(f"Image file not found: {self._image_path}")

            LOGGER.info(
                "Worker runtime env: PID=%s, PYTHONPATH=%s",
                os.getpid(),
                os.environ.get("PYTHONPATH", ""),
            )
            # 1. 确保模型可用（首次使用时下载）
            if not self._model_manager.is_model_available():
                LOGGER.info(
                    "Model not found in cache, downloading to: %s",
                    self._model_manager.model_path,
                )
                self.progress.emit("正在下载 AI 模型，请稍候…")

                def _on_progress(downloaded: int, total: int) -> None:
                    if total > 0:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        self.progress.emit(
                            f"下载模型中… {pct}%  ({mb:.1f}/{total_mb:.1f} MB)"
                        )
                    else:
                        mb = downloaded / (1024 * 1024)
                        self.progress.emit(f"下载模型中… {mb:.1f} MB")

                model_path = self._model_manager.get_model_path(
                    progress_callback=_on_progress
                )
            else:
                model_path = self._model_manager.model_path
                LOGGER.info("Using cached model: %s", model_path)
            LOGGER.info(
                "Model file check: exists=%s, size=%s bytes",
                Path(model_path).exists(),
                Path(model_path).stat().st_size if Path(model_path).exists() else "N/A",
            )

            # 2. 执行推理
            self.progress.emit("正在执行 AI 推理…")
            LOGGER.info("Initializing inference bridge with model: %s", model_path)
            bridge = SpineInferenceBridge(model_path=model_path)
            LOGGER.info("Running inference for image: %s", self._image_path)
            annotations = bridge.run_inference(self._image_path)
            LOGGER.info(
                "Inference finished successfully: image=%s, annotations=%d",
                self._image_path,
                len(annotations),
            )
            LOGGER.info("Inference worker elapsed: %.3f s", time.perf_counter() - start_ts)

            # 3. 返回结果
            self.finished.emit(annotations)

        except Exception as exc:
            traceback_text = traceback.format_exc()
            LOGGER.error(
                "Inference failed: image=%s, error=%s\n%s",
                self._image_path,
                exc,
                traceback_text,
            )
            self.error.emit(f"{type(exc).__name__}: {exc}")
