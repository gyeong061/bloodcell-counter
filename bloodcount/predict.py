"""
bloodcount/predict.py — 이미지 1장 또는 폴더를 분석해 JSON·박스 이미지·PDF 리포트를 저장

[사용 예]
  python -m bloodcount.predict --source data/images/test/BloodImage_00007.jpg
  python -m bloodcount.predict --source data/images/test --name test_batch --pdf

[산출물]  runs/predict/<name>/  (results.json, *_annotated.jpg, report.pdf)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from bloodcount.common import ROOT, resolve_path
from bloodcount.counting import CLASSES, aggregate_counts
from bloodcount.inference import PREDICT_CONFIG, CellCounter, draw_detections

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    parser = argparse.ArgumentParser(description="혈구 카운팅 추론")
    parser.add_argument("--source", type=Path, required=True, help="이미지 파일 또는 폴더")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=None, help="공통 임계값 덮어쓰기")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--pdf", action="store_true", help="PDF 리포트도 생성")
    args = parser.parse_args()

    source = resolve_path(args.source, must_exist=True)
    images = list_images(source)
    if not images:
        raise FileNotFoundError(f"이미지가 없습니다: {source}")

    counter = CellCounter(PREDICT_CONFIG, weights=args.weights)
    out_dir = ROOT / "runs" / "predict" / (args.name or source.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    results, annotated = [], []
    for path in images:
        image = Image.open(path).convert("RGB")
        result = counter.analyze(image, source=path.name, conf=args.conf)
        drawn = draw_detections(image, result.detections)
        drawn.save(out_dir / f"{path.stem}_annotated.jpg", quality=90)
        results.append(result)
        annotated.append(drawn)
        print(f"{path.name}: " + " · ".join(f"{c} {result.counts[c]}" for c in CLASSES)
              + (f"  ⚠ {len(result.flags)}건 검토" if result.flags else ""))

    summary = aggregate_counts([r.counts for r in results])
    payload = {"predicted_at": datetime.now().isoformat(timespec="seconds"),
               "model": counter.info(), "summary": summary,
               "fields": [r.to_dict() for r in results]}
    (out_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.pdf:
        from bloodcount.report import build_pdf

        pdf = build_pdf(results, annotated, sample_id=source.stem, model_name=counter.weights.name)
        (out_dir / "report.pdf").write_bytes(pdf)

    print(f"\n합계 {summary['totals']} · 시야 {summary['fields']}장 → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        sys.exit(1)
