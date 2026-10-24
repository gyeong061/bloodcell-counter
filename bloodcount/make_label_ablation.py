"""
bloodcount/make_label_ablation.py — "라벨 품질만 다른" 통제 실험용 데이터 만들기

[질문]  RBC 라벨 누락이 모델 성능을 얼마나 떨어뜨리나?

[설계]  이미지는 완전히 같고 라벨만 다른 두 데이터셋을 만듭니다.
  - 이미지   : TXL-PBC train/val에 들어 있는 BCCD 이미지 (train 101장, val 41장)
  - 라벨 A   : BCCD 원본 라벨          → data/ablation/bccd_labels/
  - 라벨 B   : TXL-PBC 재라벨          → data/ablation/txl_labels/
  - 평가     : 두 모델 모두 TXL-PBC test (재라벨 정답)로 평가
               학습 이미지가 모두 TXL train/val이라 test와 겹치지 않음 → 누수 없음

  데이터 양·이미지·학습 설정이 같으므로, 결과 차이는 라벨 품질 차이로 볼 수 있습니다.

[사용 예]
  python -m bloodcount.prepare_data           # bccd, txl 먼저 준비
  python -m bloodcount.make_label_ablation
  python -m bloodcount.train --config configs/train_ablation.yaml --data data/ablation/bccd_labels/data.yaml --name abl_bccd_labels
  python -m bloodcount.train --config configs/train_ablation.yaml --data data/ablation/txl_labels/data.yaml  --name abl_txl_labels
"""

from __future__ import annotations

import csv
import shutil

import yaml

from bloodcount.common import CLASS_NAMES, DATA_DIR, load_metadata

OUT_ROOT = DATA_DIR / "ablation"


def main() -> None:
    bccd_dir, txl_dir = DATA_DIR / "bccd", DATA_DIR / "txl"
    bccd_by_name = {r["original_name"]: r for r in load_metadata(bccd_dir).values()}
    txl_meta = load_metadata(txl_dir)
    if not bccd_by_name or not txl_meta:
        raise SystemExit("먼저 `python -m bloodcount.prepare_data`로 bccd와 txl을 준비하세요.")

    variants = {"bccd_labels": bccd_dir, "txl_labels": txl_dir}
    for variant in variants:
        shutil.rmtree(OUT_ROOT / variant, ignore_errors=True)

    rows: list[dict] = []
    for stem, row in sorted(txl_meta.items()):
        if row["source"] != "BCCD" or row["split"] not in ("train", "val"):
            continue
        orig = bccd_by_name.get(row["original_name"])
        if orig is None:
            continue
        split = row["split"]
        label_src = {"bccd_labels": bccd_dir / "labels" / orig["split"] / f"{orig['stem']}.txt",
                     "txl_labels": txl_dir / "labels" / split / f"{stem}.txt"}
        for variant in variants:
            out = OUT_ROOT / variant
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(txl_dir / "images" / split / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg")
            shutil.copy2(label_src[variant], out / "labels" / split / f"{stem}.txt")
        rows.append({"stem": stem, "split": split, "source": "BCCD",
                     "original_name": row["original_name"]})

    summary = {}
    for variant in variants:
        out = OUT_ROOT / variant
        (out / "data.yaml").write_text(yaml.safe_dump({
            "path": str(out.resolve()), "train": "images/train", "val": "images/val",
            "names": dict(CLASS_NAMES)}, sort_keys=False))
        with (out / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["stem", "split", "source", "original_name"])
            writer.writeheader()
            writer.writerows(rows)
        boxes = {name: 0 for name in CLASS_NAMES.values()}
        for txt in (out / "labels" / "train").glob("*.txt"):
            for line in txt.read_text().splitlines():
                if line.strip():
                    boxes[CLASS_NAMES[int(line.split()[0])]] += 1
        summary[variant] = {"train_images": sum(r["split"] == "train" for r in rows),
                            "val_images": sum(r["split"] == "val" for r in rows),
                            "train_boxes": boxes}
    (OUT_ROOT / "ablation_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
