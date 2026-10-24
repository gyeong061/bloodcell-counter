"""
bloodcount/evaluate.py — Test 세트 최종 평가 (탐지 지표 + 카운팅 지표)

[두 가지 관점으로 평가]
  1. 탐지 지표   : mAP50, mAP50-95, Precision, Recall (클래스별) — Ultralytics val
  2. 카운팅 지표 : 이미지별 예측 개수 vs 정답 개수의 MAE·편향·±1 일치율
     → 서비스 사용자에게 중요한 건 "몇 개를 셌는가"
  둘 다 전체, **출처(원본 데이터셋)별**, 서비스 대상과 같은 넓은 시야 도말(smear = BCCD+BCDD)로 나눠 계산하고,
  카운팅 지표에는 이미지 단위 부트스트랩 95% 신뢰구간을 붙입니다.

[신뢰도 임계값 탐색]
  카운트는 임계값에 따라 크게 달라지므로 conf 0.05~0.70에서 클래스별 카운트 MAE가 가장 낮은 값을 찾습니다.
  임계값은 val에서 고르고 test에는 그대로 적용합니다 (test로 고르면 test 점수가 낙관적으로 나옴).

[공정한 평가를 위한 제외 규칙]
  - suspect_leaks.yaml (eda.py 생성): train과 같은 촬영·거의 같은 이미지인 val/test → 제외
  - --exclude-seen-by data/bccd : 다른 데이터셋의 train/val에 쓰인 원본 이미지 → 제외
    (BCCD로 학습한 모델을 TXL-PBC test로 평가할 때 필요. TXL test 안의 BCCD 이미지 일부는
     BCCD 학습 데이터와 같은 사진이기 때문)

[사용 예]
  python -m bloodcount.evaluate --weights runs/detect/txl_yolo11s/weights/best.pt
  python -m bloodcount.evaluate --weights runs/detect/bccd_yolo11n/weights/best.pt \\
         --exclude-seen-by data/bccd --name bccd_model_on_txl
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from PIL import Image

from bloodcount.common import (CLASS_NAMES, DATA_YAML, ROOT, load_metadata, load_yaml,
                               resolve_path, seen_originals)
from bloodcount.counting import CLASSES, Detection, bootstrap_ci, count_cells, count_errors
from bloodcount.inference import PREDICT_CONFIG, CellCounter

CONF_GRID = [round(0.05 * i, 2) for i in range(1, 15)]  # 0.05 ~ 0.70
# 서비스 대상(넓은 시야 도말)과 같은 형태의 출처. WBC를 가운데 두고 자른 PBC·Raabin과 따로 봅니다.
SMEAR_SOURCES = ("BCCD", "BCDD")


def gt_counts_for(label_path: Path) -> dict[str, int]:
    """YOLO 라벨 파일에서 클래스별 정답 개수를 셉니다."""
    counts = {name: 0 for name in CLASSES}
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return counts


def select_images(data_root: Path, split: str, exclude_stems: set[str],
                  exclude_originals: set[str], meta: dict[str, dict]) -> tuple[list[Path], int]:
    """분할의 이미지 목록에서 제외 규칙을 적용합니다. (남은 목록, 제외 수) 반환."""
    kept, dropped = [], 0
    for img in sorted((data_root / "images" / split).glob("*.jpg")):
        original = meta.get(img.stem, {}).get("original_name", "")
        if img.stem in exclude_stems or (original and original in exclude_originals):
            dropped += 1
            continue
        kept.append(img)
    return kept, dropped


def collect(counter: CellCounter, images: list[Path]) -> tuple[list[list[Detection]], list[dict]]:
    """이미지들을 낮은 임계값(0.05)으로 한 번만 추론해 둡니다 (임계값별 재계산은 필터로)."""
    detections, gts = [], []
    for img_path in images:
        detections.append(counter.detect(Image.open(img_path).convert("RGB"), conf=CONF_GRID[0]))
        gts.append(gt_counts_for(img_path.parent.parent.parent / "labels" / img_path.parent.name
                                 / f"{img_path.stem}.txt"))
    return detections, gts


def counts_at(detections: list[list[Detection]], thresholds: dict[str, float]) -> list[dict]:
    """클래스별 임계값을 적용해 이미지별 개수를 다시 셉니다 (재추론 없이)."""
    return [count_cells([d for d in dets if d.confidence >= thresholds[d.class_name]])
            for dets in detections]


def tune_thresholds(detections: list[list[Detection]], gts: list[dict]) -> tuple[dict, dict]:
    """클래스마다 카운트 MAE가 가장 낮은 conf를 고릅니다. (최적값, 전체 곡선) 반환."""
    curve: dict[str, list[float]] = {name: [] for name in CLASSES}
    for conf in CONF_GRID:
        errors = count_errors(counts_at(detections, {n: conf for n in CLASSES}), gts)
        for name in CLASSES:
            curve[name].append(errors[name]["mae"])
    # 오차가 같으면 기본값(0.25)에 가까운 임계값을 고름 → 곡선이 평평할 때 극단값 방지
    best = {name: CONF_GRID[min(range(len(CONF_GRID)),
                                key=lambda i, n=name: (round(curve[n][i], 2), abs(CONF_GRID[i] - 0.25)))]
            for name in CLASSES}
    return best, curve


def plot_threshold_curve(curve: dict, best: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in CLASSES:
        ax.plot(CONF_GRID, curve[name], marker="o", ms=3, label=f"{name} (best {best[name]})")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("count MAE per image (val)")
    ax.set_title("Counting error vs threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_count_scatter(pred: list[dict], gts: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, name in zip(axes, CLASSES):
        g = [c[name] for c in gts]
        p = [c[name] for c in pred]
        hi = max(g + p + [1]) + 1
        ax.scatter(g, p, alpha=0.5)
        ax.plot([0, hi], [0, hi], "--", color="gray", lw=1)
        ax.set_title(name)
        ax.set_xlabel("ground truth")
        ax.set_xlim(-0.5, hi)
        ax.set_ylim(-0.5, hi)
    axes[0].set_ylabel("predicted")
    fig.suptitle("Predicted vs ground-truth count per image (test)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def detection_metrics(weights: Path, data_root: Path, images: list[Path], device: str,
                      out_dir: Path, tag: str) -> dict:
    """주어진 이미지 목록으로 Ultralytics val을 돌려 mAP를 계산합니다."""
    from ultralytics import YOLO

    list_file = out_dir / f"{tag}_images.txt"
    list_file.write_text("\n".join(str(p) for p in images) + "\n")
    eval_yaml = out_dir / f"{tag}_data.yaml"
    eval_yaml.write_text(yaml.safe_dump({"path": str(data_root), "train": "images/train",
                                         "val": str(list_file), "test": str(list_file),
                                         "names": dict(CLASS_NAMES)}, sort_keys=False))
    metrics = YOLO(str(weights)).val(data=str(eval_yaml), split="test", device=device,
                                     project=str(out_dir), name=f"val_{tag}", exist_ok=True,
                                     verbose=False, plots=(tag == "all"))
    box = metrics.box
    per_class = {CLASS_NAMES[int(cid)]: {
        "precision": round(float(box.p[i]), 3), "recall": round(float(box.r[i]), 3),
        "map50": round(float(box.ap50[i]), 3), "map50_95": round(float(box.ap[i]), 3)}
        for i, cid in enumerate(metrics.ap_class_index)}
    return {"images": len(images), "map50": round(float(box.map50), 4),
            "map50_95": round(float(box.map), 4), "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4), "per_class": per_class}


def main() -> None:
    parser = argparse.ArgumentParser(description="Test 세트 최종 평가")
    parser.add_argument("--weights", type=Path, default=None, help="기본: configs/predict.yaml")
    parser.add_argument("--data", type=Path, default=DATA_YAML, help="평가 데이터 (기본: TXL-PBC)")
    parser.add_argument("--exclude-seen-by", type=Path, default=None,
                        help="이 데이터셋의 train/val 원본 이미지를 평가에서 제외 (예: data/bccd)")
    parser.add_argument("--keep-suspects", action="store_true",
                        help="suspect_leaks.yaml의 의심 이미지를 제외하지 않음")
    parser.add_argument("--name", type=str, default=None, help="결과 폴더 (기본: 가중치 run 이름)")
    args = parser.parse_args()

    counter = CellCounter(PREDICT_CONFIG, weights=args.weights)
    data_yaml = resolve_path(args.data, must_exist=True,
                             hint="먼저 `python -m bloodcount.prepare_data`를 실행하세요.")
    data_root = Path(load_yaml(data_yaml)["path"])
    meta = load_metadata(data_root)
    out_dir = ROOT / "runs" / "eval" / (args.name or counter.weights.parent.parent.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    suspects_file = data_root / "suspect_leaks.yaml"
    suspects = load_yaml(suspects_file) if suspects_file.exists() and not args.keep_suspects else {}
    seen = seen_originals(resolve_path(args.exclude_seen_by)) if args.exclude_seen_by else set()

    val_imgs, val_drop = select_images(data_root, "val", set(suspects.get("val", [])), seen, meta)
    test_imgs, test_drop = select_images(data_root, "test", set(suspects.get("test", [])), seen, meta)
    print(f"평가 이미지: val {len(val_imgs)}장 (제외 {val_drop}) · test {len(test_imgs)}장 (제외 {test_drop})")

    by_source: dict[str, list[int]] = defaultdict(list)
    for i, img in enumerate(test_imgs):
        by_source[meta.get(img.stem, {}).get("source", "all")].append(i)
    if len(by_source) > 1:
        smear = sorted(i for src in SMEAR_SOURCES for i in by_source.get(src, []))
        if smear:
            by_source["smear"] = smear  # 넓은 시야 도말 (BCCD+BCDD)

    print("1/3 Test 탐지 지표 (전체 + 출처별)...")
    det = {"all": detection_metrics(counter.weights, data_root, test_imgs, counter.device, out_dir, "all")}
    if len(by_source) > 1:
        for source, idx in sorted(by_source.items()):
            det[source] = detection_metrics(counter.weights, data_root, [test_imgs[i] for i in idx],
                                            counter.device, out_dir, source)

    print("2/3 Val로 카운팅 임계값 탐색...")
    val_dets, val_gts = collect(counter, val_imgs)
    best_conf, curve = tune_thresholds(val_dets, val_gts)
    plot_threshold_curve(curve, best_conf, out_dir / "threshold_curve.png")

    print("3/3 Test 카운팅 오차...")
    test_dets, test_gts = collect(counter, test_imgs)
    pred_default = counts_at(test_dets, {n: counter.conf for n in CLASSES})
    pred_tuned = counts_at(test_dets, best_conf)
    plot_count_scatter(pred_tuned, test_gts, out_dir / "count_scatter.png")

    counting = {"all": {f"default_conf_{counter.conf}": count_errors(pred_default, test_gts),
                        "tuned_conf": count_errors(pred_tuned, test_gts),
                        "tuned_conf_ci95": bootstrap_ci(pred_tuned, test_gts)}}
    if len(by_source) > 1:
        for source, idx in sorted(by_source.items()):
            sub_pred, sub_gt = [pred_tuned[i] for i in idx], [test_gts[i] for i in idx]
            counting[source] = {"images": len(idx), "tuned_conf": count_errors(sub_pred, sub_gt)}
            if source == "smear":
                counting[source]["tuned_conf_ci95"] = bootstrap_ci(sub_pred, sub_gt)

    summary = {
        "weights": str(counter.weights.relative_to(ROOT)) if counter.weights.is_relative_to(ROOT)
        else str(counter.weights),
        "data": str(data_yaml),
        "excluded": {"val": val_drop, "test": test_drop,
                     "rules": [r for r, on in (("suspect_leaks", bool(suspects)),
                                               (f"seen_by:{args.exclude_seen_by}", bool(seen))) if on]},
        "test_images": len(test_imgs),
        "detection_test": det,
        "counting_test": counting,
        "tuned_conf_from_val": best_conf,
    }
    # 이미지별 개수 (두 모델을 같은 이미지에서 짝지어 비교할 때 사용)
    with (out_dir / "test_counts.csv").open("w", encoding="utf-8") as f:
        f.write("stem,source," + ",".join(f"gt_{c},pred_{c}" for c in CLASSES) + "\n")
        for img, gt, pred in zip(test_imgs, test_gts, pred_tuned):
            source = meta.get(img.stem, {}).get("source", "")
            f.write(f"{img.stem},{source}," + ",".join(f"{gt[c]},{pred[c]}" for c in CLASSES) + "\n")

    (out_dir / "eval_summary.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
