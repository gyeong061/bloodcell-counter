"""
bloodcount/counting.py — 탐지 결과를 "세기" 정보로 바꾸는 로직

모델(YOLO)은 박스를 내놓을 뿐이고, 서비스 가치는 이 파일에서 만들어집니다.
  - 이미지(시야) 1장의 클래스별 개수
  - 여러 시야를 합친 집계와 비율 (WBC·혈소판 / RBC 100개당)
  - 정답 대비 카운트 오차 (MAE, ±1 일치율) — mAP보다 직관적인 서비스 지표
  - 검토 필요 표시 (진단이 아니라 "사람이 다시 볼 곳"을 알려주는 용도)

ultralytics에 의존하지 않아 테스트가 빠르고, API·앱·평가 스크립트가 함께 씁니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

CLASSES: tuple[str, ...] = ("RBC", "WBC", "Platelets")


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["bbox_xyxy"] = [round(v, 1) for v in self.bbox_xyxy]
        data["confidence"] = round(self.confidence, 4)
        return data


def count_cells(detections: Iterable[Detection], classes: Sequence[str] = CLASSES) -> dict[str, int]:
    """시야 1장의 클래스별 개수. 모든 클래스를 0으로 채워 키가 항상 같도록 합니다."""
    counts = {name: 0 for name in classes}
    for det in detections:
        if det.class_name in counts:
            counts[det.class_name] += 1
    return counts


def aggregate_counts(per_field: Sequence[Mapping[str, int]],
                     classes: Sequence[str] = CLASSES) -> dict:
    """
    여러 시야(이미지)를 합칩니다.
    실제 도말 검토도 한 시야가 아니라 여러 시야를 보고 판단하므로, 합계와 시야당 평균을 함께 제공합니다.
    """
    n_fields = len(per_field)
    totals = {name: sum(int(c.get(name, 0)) for c in per_field) for name in classes}
    mean = {name: round(totals[name] / n_fields, 2) if n_fields else 0.0 for name in classes}
    return {"fields": n_fields, "totals": totals, "mean_per_field": mean,
            "ratios": cell_ratios(totals)}


def cell_ratios(counts: Mapping[str, int]) -> dict[str, float | None]:
    """RBC 100개당 WBC·혈소판 수. RBC가 0이면 계산하지 않습니다(None)."""
    rbc = counts.get("RBC", 0)
    if rbc == 0:
        return {"WBC_per_100_RBC": None, "Platelets_per_100_RBC": None}
    return {
        "WBC_per_100_RBC": round(counts.get("WBC", 0) / rbc * 100, 2),
        "Platelets_per_100_RBC": round(counts.get("Platelets", 0) / rbc * 100, 2),
    }


def review_flags(counts: Mapping[str, int], low_conf_ratio: float = 0.0,
                 min_rbc: int = 5) -> list[str]:
    """
    사람이 다시 확인하면 좋은 상황을 문장으로 알려줍니다.
    ※ 임상 판단(정상/비정상)이 아니라 '이 시야의 결과를 믿기 어려운 이유'만 다룹니다.
    """
    flags: list[str] = []
    if counts.get("RBC", 0) < min_rbc:
        flags.append(f"RBC가 {min_rbc}개 미만입니다. 초점·조명·시야 위치를 확인하세요.")
    if counts.get("WBC", 0) == 0:
        flags.append("WBC가 검출되지 않았습니다. 다른 시야를 추가로 촬영해 보세요.")
    if low_conf_ratio >= 0.3:
        flags.append(f"신뢰도 낮은 검출이 {low_conf_ratio:.0%}입니다. 결과를 직접 검토하세요.")
    return flags


def bootstrap_ci(pred: Sequence[Mapping[str, int]], gt: Sequence[Mapping[str, int]],
                 classes: Sequence[str] = CLASSES, n_boot: int = 2000, seed: int = 0,
                 tolerance: int = 1) -> dict[str, dict]:
    """
    이미지 단위 부트스트랩으로 카운팅 지표의 95% 신뢰구간을 구합니다.

    test 이미지가 적을 때 "MAE 2.5"가 얼마나 흔들릴 수 있는 값인지 함께 보여주기 위함입니다.
    반환: {클래스: {"mae": [하한, 상한], "within_tol": [하한, 상한]}}
    """
    import random

    if len(pred) != len(gt) or not pred:
        raise ValueError("예측과 정답 이미지 수가 같고 1장 이상이어야 합니다.")
    rng = random.Random(seed)
    n = len(pred)
    result: dict[str, dict] = {}
    for name in classes:
        abs_err = [abs(int(p.get(name, 0)) - int(g.get(name, 0))) for p, g in zip(pred, gt)]
        maes, tols = [], []
        for _ in range(n_boot):
            sample = [abs_err[rng.randrange(n)] for _ in range(n)]
            maes.append(sum(sample) / n)
            tols.append(sum(e <= tolerance for e in sample) / n)
        maes.sort()
        tols.sort()
        lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
        result[name] = {"mae": [round(maes[lo], 3), round(maes[hi], 3)],
                        "within_tol": [round(tols[lo], 3), round(tols[hi], 3)]}
    return result


def count_errors(pred: Sequence[Mapping[str, int]], gt: Sequence[Mapping[str, int]],
                 classes: Sequence[str] = CLASSES, tolerance: int = 1) -> dict[str, dict]:
    """
    이미지별 예측 개수와 정답 개수를 비교합니다.

    - mae         : 평균 절대 오차 (개)
    - bias        : 평균 (예측 - 정답). 음수면 덜 세는 경향
    - within_tol  : |오차| ≤ tolerance 인 이미지 비율
    - mape        : 정답이 0이 아닌 이미지에서의 평균 상대 오차
    """
    if len(pred) != len(gt):
        raise ValueError(f"예측({len(pred)})과 정답({len(gt)}) 이미지 수가 다릅니다.")
    if not pred:
        raise ValueError("비교할 이미지가 없습니다.")

    result: dict[str, dict] = {}
    for name in classes:
        diffs = [int(p.get(name, 0)) - int(g.get(name, 0)) for p, g in zip(pred, gt)]
        rel = [abs(d) / g.get(name) for d, g in zip(diffs, gt) if g.get(name, 0) > 0]
        result[name] = {
            "mae": round(sum(abs(d) for d in diffs) / len(diffs), 3),
            "bias": round(sum(diffs) / len(diffs), 3),
            "within_tol": round(sum(abs(d) <= tolerance for d in diffs) / len(diffs), 3),
            "mape": round(sum(rel) / len(rel), 3) if rel else None,
        }
    return result
