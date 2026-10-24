"""
bloodcount/train.py — YOLO11 학습

[사용 예]
  python -m bloodcount.train                                     # 본학습 (configs/train.yaml)
  python -m bloodcount.train --config configs/train_baseline.yaml # Baseline
  python -m bloodcount.train --test                              # 1 epoch 스모크 테스트
  python -m bloodcount.train --epochs 30 --name quick            # 일부 값만 바꿔 실행
  python -m bloodcount.train --seed 7 --name txl_yolo11s_s7      # 시드 반복

[산출물]  runs/detect/<name>/  (weights/best.pt, results.csv, results.png ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from bloodcount.common import (CONFIG_DIR, DATA_YAML, is_oom_error, load_yaml,
                               pick_device, resolve_path)


def build_train_kwargs(cfg: dict[str, Any], data_yaml: Path) -> dict[str, Any]:
    """설정 YAML의 hyperparameters + augmentation을 model.train() 인자로 합칩니다."""
    hyper = dict(cfg.get("hyperparameters") or {})
    aug = dict(cfg.get("augmentation") or {})
    if "project" in hyper:
        hyper["project"] = str(resolve_path(hyper["project"]))
    return {"data": str(data_yaml), "device": pick_device(cfg.get("device")), **hyper, **aug}


def train_with_oom_fallback(model: Any, kwargs: dict[str, Any], batches: list[int]) -> Any:
    """메모리 부족이 나면 batch를 줄여 다시 시도합니다. 다른 오류는 그대로 올립니다."""
    last_error: BaseException | None = None
    for batch in batches:
        print(f"\n[학습 시작] batch={batch} device={kwargs['device']}")
        try:
            return model.train(**{**kwargs, "batch": batch})
        except Exception as exc:  # noqa: BLE001 — OOM만 골라내고 나머지는 재전파
            if not is_oom_error(exc):
                raise
            print(f"[OOM] batch={batch} 실패 → 더 작은 batch로 재시도")
            last_error = exc
    raise RuntimeError("모든 batch 크기에서 메모리가 부족했습니다. imgsz를 줄여 보세요.") from last_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="혈구 탐지 YOLO11 학습")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "train.yaml")
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--epochs", type=int, default=None, help="epoch 수 덮어쓰기")
    parser.add_argument("--name", type=str, default=None, help="결과 폴더 이름 덮어쓰기")
    parser.add_argument("--seed", type=int, default=None, help="시드 덮어쓰기 (반복 실험용)")
    parser.add_argument("--test", action="store_true", help="1 epoch, 데이터 일부로 파이프라인만 확인")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(resolve_path(args.config, must_exist=True))
    data_yaml = resolve_path(args.data, must_exist=True,
                             hint="먼저 `python -m bloodcount.prepare_data`를 실행하세요.")
    kwargs = build_train_kwargs(cfg, data_yaml)

    if args.test:
        kwargs.update(epochs=1, fraction=0.25, name="smoke_test", plots=False)
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.name:
        kwargs["name"] = args.name
    if args.seed is not None:
        kwargs["seed"] = args.seed

    from ultralytics import YOLO  # 무거운 import는 실제 학습 직전에

    model_name = cfg.get("model", "yolo11s.pt")
    print(f"모델 {model_name} · 데이터 {data_yaml} · epochs {kwargs.get('epochs')}")
    batches = list(cfg.get("oom_fallback_batches") or [kwargs.get("batch", 16)])
    if kwargs.get("batch") not in batches:
        batches.insert(0, kwargs["batch"])

    train_with_oom_fallback(YOLO(model_name), kwargs, batches)
    print(f"\n완료 → {Path(kwargs['project']) / kwargs['name']}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
