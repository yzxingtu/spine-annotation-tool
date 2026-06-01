"""AI 推理桥接：调用 spine-infer SDK，将检测结果转为标注工具内部格式。

功能：
  - ModelManager: 模型下载与本地缓存管理（从阿里云 OSS 自动拉取）
  - SpineInferenceBridge: SDK Detection → OBBAnnotation 映射
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import tempfile
import time
import traceback
import urllib.request
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .models import (
    OBBAnnotation,
    Point,
    VERTEBRA_CLASSES,
    get_vertebra_class_id,
)

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

#: ONNX 模型下载地址（阿里云 OSS），部署后替换为实际 URL
MODEL_URL: str = (
    "https://yzxingtu.oss-cn-hangzhou.aliyuncs.com"
    "/spine-discern/model/spine-pose-v0.2.0-1024.onnx"
)

#: 模型文件名（与 URL 末尾一致）
MODEL_FILENAME: str = "spine-pose-v0.2.0-1024.onnx"

#: 推理输入尺寸（与模型导出时一致）
MODEL_INPUT_SIZE: int = 1024

#: SDK 类别 → 内部 class_id 起始值
# C=0(C7), T=1~12(T1~T12), L=13~17(L1~L5), S=18(S1)
_CATEGORY_CLASS_ID_START: Dict[str, int] = {
    "C": 0,    # C7
    "T": 1,    # T1
    "L": 13,   # L1
    "S": 18,   # S1
}

#: 默认每类最大检测框数（按置信度截取）
DEFAULT_MAX_PER_CATEGORY: Dict[str, int] = {
    "C": 1,
    "T": 12,
    "L": 5,
    "S": 1,
}

LOGGER = logging.getLogger("spine_annotator.inference")


def _safe_pkg_version(dist_name: str) -> str:
    """Return installed package version string for diagnostics."""
    try:
        return pkg_version(dist_name)
    except PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "unknown"


def _log_inference_environment_once() -> None:
    """Log runtime environment details helpful for DLL/import issues."""
    LOGGER.info(
        "Inference env: exe=%s, cwd=%s, frozen=%s",
        sys.executable,
        os.getcwd(),
        getattr(sys, "frozen", False),
    )
    LOGGER.info(
        "Inference env: platform=%s, release=%s, version=%s, machine=%s",
        platform.system(),
        platform.release(),
        platform.version(),
        platform.machine(),
    )
    LOGGER.info(
        "Inference deps: spine-infer=%s, onnxruntime=%s",
        _safe_pkg_version("spine-infer"),
        _safe_pkg_version("onnxruntime"),
    )
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    LOGGER.info("PATH head entries: %s", path_entries[:8])


# ---------------------------------------------------------------------------
# 模型管理
# ---------------------------------------------------------------------------

class ModelManager:
    """ONNX 模型的下载与本地缓存管理。

    缓存路径: ``~/.cache/spine-annotator/models/<MODEL_FILENAME>``
    """

    def __init__(
        self,
        model_url: str = MODEL_URL,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._model_url = model_url
        if cache_dir is None:
            if sys.platform == "darwin":
                cache_dir = Path.home() / "Library" / "Caches" / "spine-annotator" / "models"
            else:
                cache_dir = Path.home() / ".cache" / "spine-annotator" / "models"
        self._cache_dir = cache_dir
        self._model_path = self._cache_dir / MODEL_FILENAME

    @property
    def model_path(self) -> Path:
        return self._model_path

    def is_model_available(self) -> bool:
        """检查模型是否已下载到本地。"""
        return self._model_path.exists() and self._model_path.stat().st_size > 0

    def get_model_path(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """获取本地模型路径；若未缓存则自动下载。

        Args:
            progress_callback: 可选的下载进度回调 ``(downloaded_bytes, total_bytes)``。
                ``total_bytes`` 为 -1 时表示服务器未返回 Content-Length。

        Returns:
            本地模型文件路径。

        Raises:
            RuntimeError: 下载失败。
        """
        if self.is_model_available():
            LOGGER.info("Model cache hit: %s", self._model_path)
            LOGGER.info("Cached model size: %d bytes", self._model_path.stat().st_size)
            return self._model_path

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Model cache miss, downloading from %s", self._model_url)
        LOGGER.info("Model download target: %s", self._model_path)

        # 先下载到临时文件，完成后 rename，避免中断导致残留
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            dir=str(self._cache_dir), suffix=".onnx.downloading"
        )
        # 必须关闭 mkstemp 返回的文件描述符，否则 Windows 上文件被锁定
        os.close(tmp_fd)
        tmp_path = Path(tmp_path_str)

        try:
            self._download_file(self._model_url, tmp_path, progress_callback)
            tmp_path.rename(self._model_path)
            LOGGER.info("Model downloaded successfully: %s", self._model_path)
            LOGGER.info("Downloaded model size: %d bytes", self._model_path.stat().st_size)
        except Exception:
            LOGGER.error(
                "Model download failed: url=%s, tmp=%s\n%s",
                self._model_url,
                tmp_path,
                traceback.format_exc(),
            )
            # 清理临时文件
            if tmp_path.exists():
                tmp_path.unlink()
            raise

        return self._model_path

    @staticmethod
    def _download_file(
        url: str,
        dest: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """流式下载文件到 dest。"""
        LOGGER.info("Start downloading model file: %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "spine-annotator/0.2"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", -1))
            LOGGER.info("Model response content-length: %s", total)
            downloaded = 0
            chunk_size = 64 * 1024  # 64 KB

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
        LOGGER.info("Model file saved: %s", dest)


# ---------------------------------------------------------------------------
# 推理桥接
# ---------------------------------------------------------------------------

class SpineInferenceBridge:
    """调用 spine-infer SDK 执行推理，将结果映射为 OBBAnnotation。

    映射规则：
      1. 按 SDK 类别 (C/T/L/S) 分组，组内按 score 降序
      2. 每组截取 top-k（默认 C=1, T=12, L=5, S=1）
      3. 按解剖序排列：C→T→L→S，映射到内部 class_id
      4. 从 Detection.keypoints (TL/TR/BR/BL) 构建 OBBAnnotation
    """

    def __init__(
        self,
        model_path: str | Path,
        max_per_category: Optional[Dict[str, int]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        input_size: int = MODEL_INPUT_SIZE,
    ) -> None:
        _log_inference_environment_once()
        try:
            from spine_infer import SpineDetector
        except Exception:
            LOGGER.error("Failed to import spine_infer:\n%s", traceback.format_exc())
            raise

        self._max_per_category = max_per_category or DEFAULT_MAX_PER_CATEGORY.copy()

        # PyInstaller 打包后 CoreML 可能不兼容，强制使用 CPU
        device = "cpu" if getattr(sys, "frozen", False) else "auto"

        try:
            LOGGER.info(
                "Initializing detector: model=%s, backend=onnx, device=%s, input_size=%s, conf=%s, iou=%s",
                model_path,
                device,
                input_size,
                conf_threshold,
                iou_threshold,
            )
            self._detector = SpineDetector(
                weights=str(model_path),
                backend="onnx",
                device=device,
                input_size=input_size,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                num_keypoints=4,
            )
            LOGGER.info("SpineDetector initialized successfully")
        except Exception:
            LOGGER.error(
                "Failed to initialize SpineDetector:\n%s",
                traceback.format_exc(),
            )
            raise

    def run_inference(self, image_path: str) -> List[OBBAnnotation]:
        """对单张图执行推理并返回分配好 class_id 的 OBBAnnotation 列表。

        返回的列表按解剖序排列（C7, T1..T12, L1..L5, S1），
        可直接赋值给 ``ImageAnnotation.annotations``。
        """
        start_ts = time.perf_counter()
        LOGGER.info("Predict start: image=%s", image_path)
        try:
            result = self._detector.predict(image_path)
        except Exception:
            LOGGER.error("Detector predict failed:\n%s", traceback.format_exc())
            raise

        # 按 SDK 类别分组
        grouped: Dict[str, list] = {"C": [], "T": [], "L": [], "S": []}
        for det in result.detections:
            cat = det.class_name  # "C" / "T" / "L" / "S"
            if cat in grouped:
                grouped[cat].append(det)

        # 组内按 score 降序，截取 top-k
        filtered = []
        for cat in ("C", "T", "L", "S"):
            dets = sorted(grouped[cat], key=lambda d: d.score, reverse=True)
            max_k = self._max_per_category.get(cat, 99)
            dets = dets[:max_k]
            # 再按 bbox y1 升序（从上到下，对应解剖序）
            dets.sort(key=lambda d: d.y1)
            filtered.extend([(cat, d) for d in dets])
        LOGGER.info(
            "Raw detections grouped count: C=%d T=%d L=%d S=%d, filtered_total=%d",
            len(grouped["C"]),
            len(grouped["T"]),
            len(grouped["L"]),
            len(grouped["S"]),
            len(filtered),
        )

        # 映射为 OBBAnnotation
        annotations: List[OBBAnnotation] = []
        class_id_counters: Dict[str, int] = {}

        for cat, det in filtered:
            start_id = _CATEGORY_CLASS_ID_START[cat]
            offset = class_id_counters.get(cat, 0)
            internal_id = start_id + offset
            class_id_counters[cat] = offset + 1

            class_name = VERTEBRA_CLASSES.get(internal_id, f"{cat}{offset}")

            # S1 使用 bbox 构建轴对齐矩形（始终保持矩形约束）
            # 其他椎骨使用 4 个 keypoints 构建 OBB
            if cat == "S":
                x1, y1, x2, y2 = det.bbox_xyxy
                points = [
                    Point(x1, y1), Point(x2, y1),
                    Point(x2, y2), Point(x1, y2),
                ]
            elif len(det.keypoints) >= 4:
                # 从 4 个 keypoints 构建 OBB（顺时针 TL, TR, BR, BL）
                points = [
                    Point(k.x, k.y) for k in det.keypoints[:4]
                ]
            else:
                # fallback: 用 bbox_xyxy 构造轴对齐矩形
                x1, y1, x2, y2 = det.bbox_xyxy
                points = [
                    Point(x1, y1), Point(x2, y1),
                    Point(x2, y2), Point(x1, y2),
                ]

            ann = OBBAnnotation(
                class_id=internal_id,
                class_name=class_name,
                points=points,
            )
            annotations.append(ann)

        LOGGER.info("Predict done: output_annotations=%d", len(annotations))
        LOGGER.info("Predict elapsed: %.3f s", time.perf_counter() - start_ts)
        return annotations
