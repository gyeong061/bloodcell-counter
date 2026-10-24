"""
bloodcount/prepare_data.py — 데이터셋 다운로드 · YOLO 형식 통일

[지원 데이터셋]
  bccd : BCCD 원본 (VOC XML, 364장). RBC 라벨 누락이 많음 → 비교 실험용
  txl  : TXL-PBC (YOLO, 1,260장). BCCD 포함 4개 공개 데이터를 사람이 다시 라벨링 → 본 실험용
         Gan et al., Scientific Data 12, 1694 (2025), MIT License

[출력]  data/<dataset>/
  images/{train,val,test}/*.jpg   labels/{train,val,test}/*.txt
  data.yaml            YOLO 학습 설정 (path는 절대 경로)
  metadata.csv         stem, split, source, original_name  ← 출처별 평가·교차 누수 점검에 사용
  prepare_summary.yaml 분할·클래스·라벨 문제 요약

[클래스 순서]  두 데이터셋 모두 RBC=0, WBC=1, Platelets=2 로 맞춥니다.
  (TXL-PBC 원본은 WBC=0, RBC=1 이라 이름 기준으로 다시 매핑)

[사용 예]
  python -m bloodcount.prepare_data                 # 두 데이터셋 모두
  python -m bloodcount.prepare_data --dataset txl
  python -m bloodcount.prepare_data --dataset bccd --source /path/to/BCCD_Dataset
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from PIL import Image

from bloodcount.common import CLASS_NAMES, DATA_DIR, ROOT

SPLITS = ("train", "val", "test")
NAME_TO_ID = {name: idx for idx, name in CLASS_NAMES.items()}

SOURCES = {
    "bccd": {"repo": "https://github.com/Shenggan/BCCD_Dataset.git", "dir": "BCCD_Dataset"},
    "txl": {"repo": "https://github.com/lugan113/TXL-PBC_Dataset.git", "dir": "TXL-PBC_Dataset"},
}
# TXL-PBC 원본 클래스 id → 이름 (TXL-PBC/data.yaml 기준)
TXL_NAMES = {0: "WBC", 1: "RBC", 2: "Platelets"}


@dataclass
class Box:
    class_id: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class ParseReport:
    """변환 중 발견한 라벨 문제를 모아 둡니다 (EDA·README 근거)."""

    fixed: list[str] = field(default_factory=list)     # 좌표 뒤집힘·경계 밖 → 보정
    dropped: list[str] = field(default_factory=list)   # 면적 0·형식 오류 → 제외
    unknown_class: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {"fixed_boxes": len(self.fixed), "dropped_boxes": len(self.dropped),
                "unknown_classes": dict(self.unknown_class),
                "examples_fixed": self.fixed[:10], "examples_dropped": self.dropped[:10]}


# ---------------------------------------------------------------------------
# 공통
# ---------------------------------------------------------------------------

def download(dataset: str, dest: Path) -> Path:
    """데이터셋 저장소를 dest에 clone 합니다. 이미 있으면 건너뜁니다."""
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    repo = SOURCES[dataset]["repo"]
    print(f"다운로드: {repo} → {dest}")
    subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)
    return dest


def reset_split_dirs(out_dir: Path) -> None:
    """다시 실행해도 결과가 같도록 출력 폴더를 비우고 만듭니다."""
    for kind in ("images", "labels"):
        for split in SPLITS:
            folder = out_dir / kind / split
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True)


def save_as_jpg(src: Path, dest: Path) -> None:
    """
    이미지를 .jpg 로 저장합니다. TXL-PBC는 확장자가 .png지만 실제 내용은 JPEG라
    다시 인코딩하지 않고 바이트를 그대로 복사합니다(화질 손실 없음).
    """
    with Image.open(src) as im:
        fmt = im.format
        if fmt == "JPEG":
            shutil.copy2(src, dest)
        else:
            im.convert("RGB").save(dest, quality=95)


def write_outputs(out_dir: Path, meta_rows: list[dict], report: ParseReport, extra: dict) -> dict:
    """data.yaml · metadata.csv · prepare_summary.yaml 작성 후 요약 반환."""
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(out_dir.resolve()), "train": "images/train", "val": "images/val",
        "test": "images/test", "nc": len(CLASS_NAMES), "names": dict(CLASS_NAMES),
    }, allow_unicode=True, sort_keys=False))

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "split", "source", "original_name"])
        writer.writeheader()
        writer.writerows(meta_rows)

    summary: dict = {"splits": {}}
    for split in SPLITS:
        rows = [r for r in meta_rows if r["split"] == split]
        boxes: Counter = Counter()
        for r in rows:
            for line in (out_dir / "labels" / split / f"{r['stem']}.txt").read_text().splitlines():
                if line.strip():
                    boxes[CLASS_NAMES[int(line.split()[0])]] += 1
        summary["splits"][split] = {"images": len(rows), "boxes": dict(boxes),
                                    "sources": dict(Counter(r["source"] for r in rows))}
    summary["label_issues"] = report.to_dict()
    summary.update(extra)
    summary["data_yaml"] = str(data_yaml)
    (out_dir / "prepare_summary.yaml").write_text(
        yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    return summary


# ---------------------------------------------------------------------------
# BCCD (VOC XML)
# ---------------------------------------------------------------------------

def parse_voc(xml_path: Path, report: ParseReport) -> tuple[int, int, list[Box]]:
    """
    VOC XML 1개를 읽어 (width, height, boxes)를 반환합니다.
    뒤집힌 좌표는 정렬하고 경계 밖은 잘라내며, 그래도 면적이 0이면 버립니다.
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(float(size.findtext("width")))
    height = int(float(size.findtext("height")))

    boxes: list[Box] = []
    for idx, obj in enumerate(root.iter("object")):
        name = (obj.findtext("name") or "").strip()
        if name not in NAME_TO_ID:
            report.unknown_class[name] += 1
            continue
        bnd = obj.find("bndbox")
        raw = [float(bnd.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        x1, x2 = sorted((raw[0], raw[2]))
        y1, y2 = sorted((raw[1], raw[3]))
        x1, x2 = max(0.0, x1), min(float(width), x2)
        y1, y2 = max(0.0, y1), min(float(height), y2)

        tag = f"{xml_path.stem}#{idx}({name})"
        if x2 - x1 < 1 or y2 - y1 < 1:
            report.dropped.append(tag)
            continue
        if [x1, y1, x2, y2] != raw:
            report.fixed.append(tag)
        boxes.append(Box(NAME_TO_ID[name], x1, y1, x2, y2))
    return width, height, boxes


def to_yolo_line(box: Box, width: int, height: int) -> str:
    """픽셀 좌표 박스 → YOLO 한 줄 (class cx cy w h, 소수 6자리)."""
    cx = (box.xmin + box.xmax) / 2 / width
    cy = (box.ymin + box.ymax) / 2 / height
    bw = (box.xmax - box.xmin) / width
    bh = (box.ymax - box.ymin) / height
    return f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def read_official_split(bccd_dir: Path) -> dict[str, list[str]]:
    """ImageSets/Main/{train,val,test}.txt 의 공식 분할을 읽고, 분할 간 중복이 없는지 확인합니다."""
    main = bccd_dir / "ImageSets" / "Main"
    split_ids = {s: [x.strip() for x in (main / f"{s}.txt").read_text().split() if x.strip()]
                 for s in SPLITS}
    seen: dict[str, str] = {}
    for split, ids in split_ids.items():
        for stem in ids:
            if stem in seen:
                raise ValueError(f"{stem}이(가) {seen[stem]}와 {split}에 중복돼 있습니다.")
            seen[stem] = split
    return split_ids


def prepare_bccd(source: Path, out_dir: Path) -> dict:
    bccd_dir = source / "BCCD"
    report = ParseReport()
    meta: list[dict] = []
    reset_split_dirs(out_dir)
    for split, stems in read_official_split(bccd_dir).items():
        for stem in stems:
            image_path = bccd_dir / "JPEGImages" / f"{stem}.jpg"
            xml_path = bccd_dir / "Annotations" / f"{stem}.xml"
            if not image_path.exists() or not xml_path.exists():
                print(f"[경고] {stem}: 이미지 또는 XML 없음 → 건너뜀")
                continue
            width, height, boxes = parse_voc(xml_path, report)
            shutil.copy2(image_path, out_dir / "images" / split / image_path.name)
            lines = [to_yolo_line(b, width, height) for b in boxes]
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            meta.append({"stem": stem, "split": split, "source": "BCCD",
                         "original_name": image_path.name})
    return write_outputs(out_dir, meta, report, {"dataset": "BCCD (original labels)"})


# ---------------------------------------------------------------------------
# TXL-PBC (YOLO, 클래스 순서 다름)
# ---------------------------------------------------------------------------

def remap_txl_label(text: str, tag: str, report: ParseReport) -> list[str]:
    """
    TXL-PBC 라벨을 우리 클래스 순서로 바꾸고 형식을 검증합니다.
    - 5개 값이 아니거나, 좌표가 0~1 밖이거나, 폭·높이가 0이면 제외
    """
    lines = []
    for idx, line in enumerate(text.splitlines()):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            report.dropped.append(f"{tag}#{idx}(형식)")
            continue
        cid = int(float(parts[0]))
        name = TXL_NAMES.get(cid)
        if name is None:
            report.unknown_class[str(cid)] += 1
            continue
        cx, cy, w, h = map(float, parts[1:])
        if w <= 0 or h <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1):
            report.dropped.append(f"{tag}#{idx}({name})")
            continue
        lines.append(f"{NAME_TO_ID[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def read_txl_metadata(txl_dir: Path) -> dict[str, dict]:
    """metadata_file.xlsx → {새 파일 stem: {source, original_name}}"""
    import pandas as pd  # openpyxl 필요

    df = pd.read_excel(txl_dir / "metadata_file.xlsx")
    return {Path(r.New_Image_Name).stem: {"source": r.Source_Dataset,
                                          "original_name": r.Original_Image_Name}
            for r in df.itertuples()}


def prepare_txl(source: Path, out_dir: Path) -> dict:
    txl_dir = source / "TXL-PBC"
    meta_map = read_txl_metadata(txl_dir)
    report = ParseReport()
    meta: list[dict] = []
    reset_split_dirs(out_dir)
    for split in SPLITS:
        for label_path in sorted((txl_dir / "labels" / split).glob("*.txt")):
            stem = label_path.stem
            candidates = list((txl_dir / "images" / split).glob(f"{stem}.*"))
            if not candidates:
                print(f"[경고] {stem}: 이미지 없음 → 건너뜀")
                continue
            save_as_jpg(candidates[0], out_dir / "images" / split / f"{stem}.jpg")
            lines = remap_txl_label(label_path.read_text(), stem, report)
            (out_dir / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""))
            info = meta_map.get(stem, {"source": "unknown", "original_name": ""})
            meta.append({"stem": stem, "split": split, **info})
    return write_outputs(out_dir, meta, report, {
        "dataset": "TXL-PBC (re-annotated)",
        "citation": "Gan, Li & Wang. Scientific Data 12, 1694 (2025). doi:10.1038/s41597-025-05980-z"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PREPARERS = {"bccd": prepare_bccd, "txl": prepare_txl}


def run(dataset: str, source: Path | None = None) -> dict:
    src = download(dataset, source or ROOT / "external" / SOURCES[dataset]["dir"])
    summary = PREPARERS[dataset](src, DATA_DIR / dataset)
    print(f"\n=== {dataset} 변환 완료 ===")
    for split, info in summary["splits"].items():
        print(f"{split:5s}: {info['images']:4d}장  {info['boxes']}  출처 {info['sources']}")
    issues = summary["label_issues"]
    print(f"라벨 보정 {issues['fixed_boxes']}개 · 제외 {issues['dropped_boxes']}개 "
          f"· 알 수 없는 클래스 {issues['unknown_classes'] or '없음'}")
    print(f"data.yaml: {summary['data_yaml']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터셋 다운로드 · YOLO 변환")
    parser.add_argument("--dataset", choices=["bccd", "txl", "all"], default="all")
    parser.add_argument("--source", type=Path, default=None,
                        help="이미 받아 둔 데이터셋 저장소 폴더 (--dataset 하나일 때만)")
    args = parser.parse_args()
    if args.source and args.dataset == "all":
        parser.error("--source는 --dataset bccd 또는 txl 과 함께 쓰세요.")
    for name in (["bccd", "txl"] if args.dataset == "all" else [args.dataset]):
        run(name, args.source.resolve() if args.source else None)


if __name__ == "__main__":
    main()
