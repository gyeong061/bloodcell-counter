"""
bloodcount/common.py — 여러 스크립트가 함께 쓰는 공용 함수

원 프로젝트(wind-turbine-yolo)에서는 load_yaml, resolve_weights, is_oom_error가
train/val/predict/app에 각각 복사돼 있었습니다. 여기서는 한 곳에 모아 중복을 없앴습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 프로젝트 루트 (bloodcount/ 의 상위 폴더)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
# 기본 데이터셋 = 다시 라벨링된 TXL-PBC. BCCD 원본은 비교 실험용 (data/bccd/data.yaml)
DATA_YAML = DATA_DIR / "txl" / "data.yaml"
CONFIG_DIR = ROOT / "configs"

# 데이터셋 클래스 — prepare_data.py가 data.yaml을 만들 때도 이 순서를 사용
CLASS_NAMES: dict[int, str] = {0: "RBC", 1: "WBC", 2: "Platelets"}


def load_yaml(path: Path | str) -> dict[str, Any]:
    """YAML 파일을 dict로 읽습니다. 파일이 없으면 이유를 알려주는 오류를 냅니다."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path: Path | str, must_exist: bool = False, hint: str = "") -> Path:
    """
    상대 경로를 프로젝트 루트 기준 절대 경로로 바꿉니다.

    실행 위치(cwd)에 따라 runs/ 가 엉뚱한 곳에 생기는 문제를 막기 위함입니다.
    """
    path = Path(path)
    resolved = (path if path.is_absolute() else ROOT / path).resolve()
    if must_exist and not resolved.exists():
        message = f"경로를 찾을 수 없습니다: {resolved}"
        if hint:
            message += f"\n{hint}"
        raise FileNotFoundError(message)
    return resolved


def load_class_names(data_yaml: Path = DATA_YAML) -> dict[int, str]:
    """data.yaml의 names를 {id: name}으로 반환합니다. 없으면 기본 클래스를 씁니다."""
    if not data_yaml.exists():
        return dict(CLASS_NAMES)
    names = load_yaml(data_yaml).get("names", CLASS_NAMES)
    if isinstance(names, list):
        return dict(enumerate(names))
    return {int(k): str(v) for k, v in names.items()}


def load_metadata(data_root: Path) -> dict[str, dict]:
    """prepare_data.py가 만든 metadata.csv → {stem: {split, source, original_name}}. 없으면 빈 dict."""
    import csv

    path = Path(data_root) / "metadata.csv"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["stem"]: row for row in csv.DictReader(f)}


def seen_originals(other_root: Path, splits: tuple[str, ...] = ("train", "val")) -> set[str]:
    """
    다른 데이터셋의 train/val에 쓰인 원본 이미지 이름 집합.

    예) BCCD로 학습한 모델을 TXL-PBC test로 평가할 때, TXL test 안의 BCCD 이미지 중
        BCCD 학습에 이미 쓰인 것을 빼야 공정합니다 (교차 데이터셋 누수 방지).
    """
    return {row["original_name"] for row in load_metadata(other_root).values()
            if row["split"] in splits and row["original_name"]}


def is_oom_error(exc: BaseException) -> bool:
    """CUDA/MPS 메모리 부족 예외인지 메시지로 판별합니다."""
    message = str(exc).lower()
    return any(key in message for key in ("out of memory", "oom", "mps backend"))


def pick_device(preferred: str | None = None) -> str:
    """
    사용 가능한 장치를 고릅니다. 설정이 'auto'이거나 비어 있으면
    CUDA(Colab) → MPS(Mac) → CPU 순서로 선택합니다.
    """
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
