"""
bloodcount/inference.py — 추론 엔진 (CLI · Streamlit · FastAPI 공용)

가중치를 한 번만 불러와 재사용하고, 이미지 1장(시야)마다
  탐지 목록 → 클래스별 개수 → 검토 표시 → 박스가 그려진 이미지
를 한 번에 돌려줍니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from bloodcount.common import CONFIG_DIR, load_yaml, pick_device, resolve_path
from bloodcount.counting import CLASSES, Detection, count_cells, review_flags

PREDICT_CONFIG = CONFIG_DIR / "predict.yaml"
BOX_COLORS = {"RBC": (209, 73, 91), "WBC": (61, 90, 128), "Platelets": (237, 174, 73)}


@dataclass
class FieldResult:
    """시야(이미지) 1장의 분석 결과."""

    source: str
    width: int
    height: int
    detections: list[Detection]
    counts: dict[str, int]
    low_conf_ratio: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "image_width": self.width,
            "image_height": self.height,
            "counts": self.counts,
            "low_conf_ratio": round(self.low_conf_ratio, 3),
            "flags": self.flags,
            "detections": [d.to_dict() for d in self.detections],
        }


def draw_detections(image: Image.Image, detections: list[Detection],
                    hidden: set[str] | None = None, width: int = 2) -> Image.Image:
    """
    클래스별 색 박스를 그립니다. 혈구가 빽빽해 글자 라벨은 생략하고 색으로 구분합니다.
    hidden에 넣은 클래스는 그리지 않습니다 (예: RBC를 숨기고 WBC만 보기).
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    hidden = hidden or set()
    for det in detections:
        if det.class_name in hidden:
            continue
        color = BOX_COLORS.get(det.class_name, (255, 255, 255))
        draw.rectangle(det.bbox_xyxy, outline=color, width=width)
    return canvas


class CellCounter:
    """YOLO 가중치를 감싼 혈구 카운터."""

    def __init__(self, config_path: Path = PREDICT_CONFIG, weights: Path | str | None = None) -> None:
        cfg = load_yaml(config_path)
        params = cfg.get("parameters", {})
        self.weights = resolve_path(
            weights or cfg["weights"], must_exist=True,
            hint="먼저 `python -m bloodcount.train`으로 학습하거나 --weights 경로를 지정하세요.")
        self.device = pick_device(cfg.get("device"))
        self.conf = float(params.get("conf", 0.25))
        self.iou = float(params.get("iou", 0.5))
        self.imgsz = int(params.get("imgsz", 640))
        self.max_det = int(params.get("max_det", 300))
        self.low_conf_threshold = float(cfg.get("low_conf_threshold", 0.4))
        # evaluate.py가 val 세트에서 찾은 클래스별 최적 임계값 (없으면 공통 conf 사용)
        self.class_conf: dict[str, float] = {
            str(k): float(v) for k, v in (cfg.get("class_conf") or {}).items()}

        from ultralytics import YOLO

        self.model = YOLO(str(self.weights))
        self.class_names: dict[int, str] = {int(k): str(v) for k, v in self.model.names.items()}

    def info(self) -> dict[str, Any]:
        return {"weights": self.weights.name, "device": self.device, "conf": self.conf,
                "iou": self.iou, "imgsz": self.imgsz, "classes": self.class_names}

    def _raw_predict(self, image: Image.Image | np.ndarray | str, conf: float) -> Any:
        return self.model.predict(source=image, device=self.device, conf=conf, iou=self.iou,
                                  imgsz=self.imgsz, max_det=self.max_det, verbose=False)[0]

    def detect(self, image: Image.Image | np.ndarray | str, conf: float | None = None) -> list[Detection]:
        """탐지 목록만 반환합니다 (평가 스크립트용)."""
        result = self._raw_predict(image, self.conf if conf is None else conf)
        return self._to_detections(result)

    def _to_detections(self, result: Any) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        return [
            Detection(self.class_names.get(int(c), str(int(c))), float(s), tuple(map(float, xyxy)))
            for xyxy, s, c in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(),
                                  boxes.cls.cpu().tolist())
        ]

    def analyze(self, image: Image.Image, source: str = "upload", conf: float | None = None) -> FieldResult:
        """시야 1장 분석: 탐지 → 개수 → 검토 표시."""
        image = image.convert("RGB")
        if conf is not None:  # 사용자가 슬라이더로 직접 정한 값이 우선
            detections = self.detect(image, conf)
        else:
            thresholds = {name: self.class_conf.get(name, self.conf) for name in CLASSES}
            detections = [d for d in self.detect(image, min(thresholds.values()))
                          if d.confidence >= thresholds.get(d.class_name, self.conf)]
        counts = count_cells(detections, CLASSES)
        low = sum(d.confidence < self.low_conf_threshold for d in detections)
        low_ratio = low / len(detections) if detections else 0.0
        return FieldResult(source=source, width=image.width, height=image.height,
                           detections=detections, counts=counts, low_conf_ratio=low_ratio,
                           flags=review_flags(counts, low_ratio))
