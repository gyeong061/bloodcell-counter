"""
bloodcount/eda.py — 데이터 탐색 (학습 전 데이터 특성·품질 점검)

[확인하는 것]
  1. 클래스 분포         : 분할별 RBC·WBC·혈소판 박스 수
  2. 이미지당 개수 분포   : 카운팅 문제의 난이도
  3. 박스 크기           : 클래스별 크기 차이 (혈소판은 작음)
  4. 출처별 특성 (TXL)   : 4개 원본 데이터의 이미지 크기·이미지당 개수 차이
  5. 분할 간 유사 이미지  : 거의 같은 사진이 train/test에 나뉘면 점수가 부풀려짐
  6. 라벨 비교 (bccd+txl): 같은 BCCD 이미지에서 원본 라벨 vs 재라벨 개수 비교
                          → RBC 라벨 누락을 숫자로 확인

[산출물]  runs/eda/<dataset>/  — 그래프 png + eda_summary.yaml

[사용 예]
  python -m bloodcount.eda                  # 기본: txl (+ bccd가 있으면 라벨 비교)
  python -m bloodcount.eda --dataset bccd
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없는 환경(Colab·서버)에서도 저장 가능
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from bloodcount.common import CLASS_NAMES, DATA_DIR, ROOT, load_metadata

OUT_ROOT = ROOT / "runs" / "eda"
SPLITS = ("train", "val", "test")
COLORS = {"RBC": "#d1495b", "WBC": "#3d5a80", "Platelets": "#edae49"}

Labels = dict[str, dict[str, list[tuple[int, float, float]]]]  # split → stem → [(cls, w, h)]


def load_labels(data_dir: Path) -> Labels:
    """{split: {image_stem: [(class_id, w, h), ...]}} 형태로 라벨을 읽습니다."""
    result: Labels = {}
    for split in SPLITS:
        per_image: dict[str, list[tuple[int, float, float]]] = {}
        for txt in sorted((data_dir / "labels" / split).glob("*.txt")):
            rows = []
            for line in txt.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    rows.append((int(parts[0]), float(parts[3]), float(parts[4])))
            per_image[txt.stem] = rows
        result[split] = per_image
    return result


def class_counts(rows: list[tuple[int, float, float]]) -> dict[str, int]:
    counter = Counter(CLASS_NAMES[r[0]] for r in rows)
    return {name: counter.get(name, 0) for name in CLASS_NAMES.values()}


# ---------------------------------------------------------------------------
# 유사 이미지 (누수) 점검
# ---------------------------------------------------------------------------

def average_hash(path: Path, size: int = 16) -> np.ndarray:
    """간단한 지각 해시(aHash): 축소 흑백 이미지가 평균보다 밝은지 여부의 비트 배열."""
    img = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    return (arr > arr.mean()).flatten()


def find_cross_split_duplicates(data_dir: Path, max_distance: int = 30) -> list[dict]:
    """
    서로 다른 분할에 속한 이미지 중 해시 거리가 가까운(거의 같은) 쌍을 찾습니다.
    16x16 해시(256비트)에서 거리 30 이하 ≈ 88% 이상 비트가 같은 이미지.
    서로 다른 이미지 쌍은 대부분 거리 60 이상이고, 직접 눈으로 확인한 같은 세포·같은 시야 쌍은
    13~32 사이였습니다. 그래서 30을 기준으로 쓰고, 이름 규칙 기반 점검(capture_group)과 함께 봅니다.
    """
    names, splits, hashes = [], [], []
    for split in SPLITS:
        for img in sorted((data_dir / "images" / split).glob("*.jpg")):
            names.append(img.stem)
            splits.append(split)
            hashes.append(average_hash(img))
    if not hashes:
        return []
    bits = np.stack(hashes).astype(np.uint8)
    split_arr = np.array(splits)
    pairs = []
    for i in range(len(bits) - 1):
        dist = np.count_nonzero(bits[i + 1:] != bits[i], axis=1)
        for j in np.nonzero((dist <= max_distance) & (split_arr[i + 1:] != split_arr[i]))[0]:
            k = i + 1 + int(j)
            pairs.append({"a": f"{splits[i]}/{names[i]}", "b": f"{splits[k]}/{names[k]}",
                          "distance": int(dist[j])})
    return sorted(pairs, key=lambda p: p["distance"])


# ---------------------------------------------------------------------------
# 그래프
# ---------------------------------------------------------------------------

def plot_class_distribution(labels: Labels, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    x = np.arange(len(SPLITS))
    for offset, (cid, name) in enumerate(CLASS_NAMES.items()):
        counts = [sum(1 for rows in labels[s].values() for r in rows if r[0] == cid) for s in SPLITS]
        bars = ax.bar(x + (offset - 1) * width, counts, width, label=name, color=COLORS[name])
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, SPLITS)
    ax.set_ylabel("boxes")
    ax.set_title("Class distribution by split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_counts_per_image(labels: Labels, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (cid, name) in zip(axes, CLASS_NAMES.items()):
        counts = [sum(1 for r in rows if r[0] == cid)
                  for split in SPLITS for rows in labels[split].values()]
        ax.hist(counts, bins=range(0, max(counts) + 2), color=COLORS[name], edgecolor="white")
        ax.set_title(f"{name} per image (mean {np.mean(counts):.1f})")
        ax.set_xlabel("count")
    axes[0].set_ylabel("images")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_box_sizes(labels: Labels, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    for cid, name in CLASS_NAMES.items():
        pts = [(r[1], r[2]) for s in SPLITS for rows in labels[s].values() for r in rows if r[0] == cid]
        if pts:
            ws, hs = zip(*pts)
            ax.scatter(ws, hs, s=5, alpha=0.3, label=name, color=COLORS[name])
    ax.set_xlabel("width (normalized)")
    ax.set_ylabel("height (normalized)")
    ax.set_title("Box size by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_by_source(by_source: dict[str, dict], out: Path) -> None:
    """출처별 이미지당 평균 개수."""
    sources = list(by_source)
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(len(sources))
    for offset, name in enumerate(CLASS_NAMES.values()):
        vals = [by_source[s]["mean_per_image"][name] for s in sources]
        bars = ax.bar(x + (offset - 1) * width, vals, width, label=name, color=COLORS[name])
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(x, [f"{s}\n({by_source[s]['images']} imgs)" for s in sources])
    ax.set_ylabel("mean count per image")
    ax.set_title("Cells per image by source dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 출처별 요약 · 라벨 비교
# ---------------------------------------------------------------------------

def summarize_by_source(data_dir: Path, labels: Labels) -> dict[str, dict]:
    meta = load_metadata(data_dir)
    if not meta:
        return {}
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split in SPLITS:
        for stem in labels[split]:
            groups[meta.get(stem, {}).get("source", "unknown")].append((split, stem))
    result = {}
    for source, items in sorted(groups.items()):
        counts = [class_counts(labels[s][stem]) for s, stem in items]
        sizes = Counter()
        for s, stem in items[:50]:  # 크기는 표본 50장으로 충분
            with Image.open(data_dir / "images" / s / f"{stem}.jpg") as im:
                sizes[f"{im.width}x{im.height}"] += 1
        result[source] = {
            "images": len(items),
            "splits": dict(Counter(s for s, _ in items)),
            "mean_per_image": {n: round(float(np.mean([c[n] for c in counts])), 2)
                               for n in CLASS_NAMES.values()},
            "image_sizes_sample": dict(sizes.most_common(3)),
        }
    return result


def capture_group(source: str, original_name: str) -> str:
    """
    같은 촬영에서 나온 이미지끼리 묶는 키.
    Raabin-WBC는 한 촬영에서 세포 여러 개를 잘라 '<촬영ID>_<번호>.jpg'로 저장하므로
    마지막 '_' 앞부분이 같으면 같은 촬영입니다. 다른 출처는 파일 하나가 한 그룹.
    """
    stem = original_name.rsplit(".", 1)[0]
    if source == "Raabin-WBC" and "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    return f"{source}:{stem}"


def find_suspect_leaks(data_dir: Path, duplicates: list[dict]) -> dict[str, list[str]]:
    """
    val/test 이미지 중 train과 같은 촬영이거나 거의 같은 이미지인 것을 모읍니다.
    결과는 data/<dataset>/suspect_leaks.yaml 로 저장되고 evaluate.py가 평가에서 제외합니다.
    """
    meta = load_metadata(data_dir)
    suspects: dict[str, set[str]] = {"val": set(), "test": set()}

    train_groups = {capture_group(r["source"], r["original_name"])
                    for r in meta.values() if r["split"] == "train"}
    for stem, r in meta.items():
        if r["split"] in suspects and capture_group(r["source"], r["original_name"]) in train_groups:
            suspects[r["split"]].add(stem)

    for pair in duplicates:
        (sa, a), (sb, b) = pair["a"].split("/"), pair["b"].split("/")
        if "train" in (sa, sb):
            split, stem = (sb, b) if sa == "train" else (sa, a)
            suspects[split].add(stem)
        else:  # val ↔ test: test 쪽을 의심 목록에
            split, stem = (sa, a) if sa == "test" else (sb, b)
            suspects[split].add(stem)
    return {k: sorted(v) for k, v in suspects.items()}


def compare_relabel(bccd_dir: Path, txl_dir: Path, out: Path) -> dict | None:
    """
    같은 BCCD 이미지에서 원본 라벨(bccd)과 재라벨(txl)의 개수를 비교합니다.
    TXL-PBC metadata의 original_name으로 두 데이터셋의 이미지를 짝지읍니다.
    """
    bccd_meta = {row["original_name"]: row for row in load_metadata(bccd_dir).values()}
    txl_meta = load_metadata(txl_dir)
    pairs = []
    for stem, row in txl_meta.items():
        if row["source"] != "BCCD" or row["original_name"] not in bccd_meta:
            continue
        b = bccd_meta[row["original_name"]]
        orig = (bccd_dir / "labels" / b["split"] / f"{b['stem']}.txt").read_text().split("\n")
        new = (txl_dir / "labels" / row["split"] / f"{stem}.txt").read_text().split("\n")
        to_rows = lambda lines: [(int(x.split()[0]), 0.0, 0.0) for x in lines if x.strip()]
        pairs.append((class_counts(to_rows(orig)), class_counts(to_rows(new))))
    if not pairs:
        return None

    names = list(CLASS_NAMES.values())
    totals_orig = {n: sum(p[0][n] for p in pairs) for n in names}
    totals_new = {n: sum(p[1][n] for p in pairs) for n in names}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(names))
    b1 = axes[0].bar(x - 0.2, [totals_orig[n] for n in names], 0.4, label="BCCD original", color="#9aa5b1")
    b2 = axes[0].bar(x + 0.2, [totals_new[n] for n in names], 0.4, label="TXL-PBC re-annotated",
                     color="#1e3a5f")
    axes[0].bar_label(b1, fontsize=8)
    axes[0].bar_label(b2, fontsize=8)
    axes[0].set_xticks(x, names)
    axes[0].set_title(f"Same {len(pairs)} BCCD images: total labels")
    axes[0].legend()
    o = [p[0]["RBC"] for p in pairs]
    n = [p[1]["RBC"] for p in pairs]
    hi = max(o + n) + 2
    axes[1].scatter(o, n, alpha=0.6, color=COLORS["RBC"])
    axes[1].plot([0, hi], [0, hi], "--", color="gray", lw=1)
    axes[1].set_xlabel("RBC labels — original")
    axes[1].set_ylabel("RBC labels — re-annotated")
    axes[1].set_title("RBC per image (above line = missing in original)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)

    return {
        "matched_images": len(pairs),
        "totals_original": totals_orig,
        "totals_reannotated": totals_new,
        "rbc_added_per_image_mean": round(float(np.mean(np.array(n) - np.array(o))), 2),
        "images_with_more_rbc": int(sum(b > a for a, b in zip(o, n))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="데이터 탐색 (EDA)")
    parser.add_argument("--dataset", choices=["txl", "bccd"], default="txl")
    args = parser.parse_args()

    data_dir = DATA_DIR / args.dataset
    out = OUT_ROOT / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(data_dir)
    if not any(labels.values()):
        raise SystemExit(f"{data_dir}에 라벨이 없습니다. 먼저 "
                         f"`python -m bloodcount.prepare_data --dataset {args.dataset}`를 실행하세요.")

    plot_class_distribution(labels, out / "class_distribution.png")
    plot_counts_per_image(labels, out / "counts_per_image.png")
    plot_box_sizes(labels, out / "box_sizes.png")

    summary: dict = {"dataset": args.dataset, "classes": {}}
    for cid, name in CLASS_NAMES.items():
        per_image = [sum(1 for r in rows if r[0] == cid) for s in SPLITS for rows in labels[s].values()]
        sizes = [max(r[1], r[2]) for s in SPLITS for rows in labels[s].values() for r in rows if r[0] == cid]
        summary["classes"][name] = {
            "boxes": int(sum(per_image)),
            "mean_per_image": round(float(np.mean(per_image)), 2),
            "max_per_image": int(max(per_image)),
            "images_with_zero": int(sum(1 for c in per_image if c == 0)),
            "median_box_side": round(float(np.median(sizes)), 3) if sizes else None,
        }

    by_source = summarize_by_source(data_dir, labels)
    if by_source:
        summary["by_source"] = by_source
        plot_by_source(by_source, out / "by_source.png")

    duplicates = find_cross_split_duplicates(data_dir)
    summary["duplicates_across_splits"] = len(duplicates)
    summary["duplicate_examples"] = duplicates[:10]

    suspects = find_suspect_leaks(data_dir, duplicates)
    (data_dir / "suspect_leaks.yaml").write_text(yaml.safe_dump(suspects, sort_keys=False))
    summary["suspect_leaks"] = {k: len(v) for k, v in suspects.items()}

    if args.dataset == "txl" and (DATA_DIR / "bccd" / "metadata.csv").exists():
        relabel = compare_relabel(DATA_DIR / "bccd", data_dir, out / "relabel_comparison.png")
        if relabel:
            summary["relabel_vs_bccd"] = relabel

    (out / "eda_summary.yaml").write_text(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    print(yaml.safe_dump(summary, allow_unicode=True, sort_keys=False))
    print(f"그래프 저장: {out}")


if __name__ == "__main__":
    main()
