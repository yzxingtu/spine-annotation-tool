"""异步推理 Worker：在后台线程执行模型下载 + 推理，避免阻塞 UI。"""

from __future__ import annotations

from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from ..core.inference import ModelManager, SpineInferenceBridge
from ..core.models import OBBAnnotation


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
            # 1. 确保模型可用（首次使用时下载）
            if not self._model_manager.is_model_available():
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

            # 2. 执行推理
            self.progress.emit("正在执行 AI 推理…")
            bridge = SpineInferenceBridge(model_path=model_path)
            annotations = bridge.run_inference(self._image_path)

            # 3. 返回结果
            self.finished.emit(annotations)

        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")
