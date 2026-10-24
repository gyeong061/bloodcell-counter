"""
bloodcount/compare_runs.py — 평가 결과를 모아 README에 붙일 비교 표 만들기

runs/eval/<이름>/eval_summary.yaml 들을 읽어, 이름 끝의 `_s<시드>`를 떼고 같은 실험끼리 묶습니다.
  - 시드가 여러 개면  평균 ± 표준편차  (시드에 따라 얼마나 흔들리는지)
  - 시드가 하나면     값 [95% 신뢰구간]  (test 이미지 표본에 따라 얼마나 흔들리는지)

[사용 예]
  python -m bloodcount.compare_runs                          # 모든 실험
  python -m bloodcount.compare_runs --only abl_ txl_         # 이름이 이걸로 시작하는 실험만
  python -m bloodcount.compare_runs --paired abl_bccd_labels_s42 abl_txl_labels_s42
      # 같은 test 이미지에서 두 모델의 개수 오차 차이 (짝지은 부트스트랩 95% 신뢰구간)

[산출물]  표준 출력 + runs/eval/comparison.md
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from bloodcount.common import ROOT

EVAL_DIR = ROOT / "runs" / "eval"
SEED_SUFFIX = re.compile(r"^(?P<group>.+)_s(?P<seed>\d+)$")


def load_runs(eval_dir: Path = EVAL_DIR) -> dict[str, list[tuple[str, dict]]]:
    """{실험 그룹: [(시드 또는 '-', summary), ...]}"""
    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for path in sorted(eval_dir.glob("*/eval_summary.yaml")):
        name = path.parent.name
        match = SEED_SUFFIX.match(name)
        group, seed = (match["group"], match["seed"]) if match else (name, "-")
        groups[group].append((seed, yaml.safe_load(path.read_text())))
    return dict(groups)


def extract(summary: dict) -> dict[str, float | None]:
    """요약 하나에서 표에 쓸 숫자만 꺼냅니다."""
    det = summary["detection_test"]
    cnt = summary["counting_test"]
    tuned = cnt["all"]["tuned_conf"]
    smear = cnt.get("smear", {}).get("tuned_conf")
    return {
        "mAP50": det["all"]["map50"],
        "mAP50-95": det["all"]["map50_95"],
        "smear mAP50": det.get("smear", {}).get("map50"),
        "RBC MAE": tuned["RBC"]["mae"],
        "RBC ±1": tuned["RBC"]["within_tol"],
        "WBC MAE": tuned["WBC"]["mae"],
        "Plt MAE": tuned["Platelets"]["mae"],
        "smear RBC MAE": smear["RBC"]["mae"] if smear else None,
    }


def ci_of(summary: dict, key: str) -> list[float] | None:
    """단일 시드일 때 표에 붙일 95% 신뢰구간."""
    cnt = summary["counting_test"]
    table = {
        "RBC MAE": ("all", "RBC", "mae"), "RBC ±1": ("all", "RBC", "within_tol"),
        "WBC MAE": ("all", "WBC", "mae"), "Plt MAE": ("all", "Platelets", "mae"),
        "smear RBC MAE": ("smear", "RBC", "mae"),
    }
    if key not in table:
        return None
    group, cls, metric = table[key]
    return cnt.get(group, {}).get("tuned_conf_ci95", {}).get(cls, {}).get(metric)


def fmt(key: str, values: list[float], ci: list[float] | None) -> str:
    pct = key.endswith("±1")
    show = (lambda v: f"{v * 100:.1f}%") if pct else (lambda v: f"{v:.3f}" if "mAP" in key else f"{v:.2f}")
    if len(values) > 1:
        return f"{show(statistics.mean(values))} ± {show(statistics.stdev(values))}"
    text = show(values[0])
    if ci:
        text += f" [{show(ci[0])}–{show(ci[1])}]"
    return text


def build_table(groups: dict[str, list[tuple[str, dict]]], only: list[str] | None = None) -> str:
    keys = ["mAP50", "mAP50-95", "smear mAP50", "RBC MAE", "RBC ±1", "WBC MAE", "Plt MAE", "smear RBC MAE"]
    header = "| 실험 | 시드 | test 이미지 | " + " | ".join(keys) + " |"
    lines = [header, "|" + "---|" * (len(keys) + 3)]
    for group, runs in sorted(groups.items()):
        if only and not any(group.startswith(prefix) for prefix in only):
            continue
        rows = [extract(s) for _, s in runs]
        seeds = ", ".join(seed for seed, _ in runs)
        cells = []
        for key in keys:
            values = [r[key] for r in rows if r[key] is not None]
            if not values:
                cells.append("–")
                continue
            ci = ci_of(runs[0][1], key) if len(values) == 1 else None
            cells.append(fmt(key, values, ci))
        lines.append(f"| {group} | {seeds} | {runs[0][1]['test_images']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("개수 지표(MAE·±1)는 val에서 고른 클래스별 임계값 기준. "
                 "시드가 여러 개면 평균 ± 표준편차, 하나면 값 [이미지 부트스트랩 95% 신뢰구간]. "
                 "smear = 넓은 시야 도말(BCCD+BCDD) test 이미지만.")
    return "\n".join(lines)


def read_counts(run: str) -> dict[str, dict]:
    """evaluate.py가 저장한 이미지별 개수 → {stem: row}"""
    import csv

    path = EVAL_DIR / run / "test_counts.csv"
    if not path.exists():
        raise SystemExit(f"{path}가 없습니다. 해당 실험을 evaluate.py로 다시 평가하세요.")
    with path.open(encoding="utf-8") as f:
        return {row["stem"]: row for row in csv.DictReader(f)}


def paired_diff(run_a: str, run_b: str, n_boot: int = 2000, seed: int = 0,
                subsets: tuple[str, ...] = ("all", "smear")) -> str:
    """
    같은 이미지에서 |오차|(B) − |오차|(A) 의 평균과 95% 신뢰구간.
    음수면 B가 더 정확. 구간이 0을 포함하면 차이가 있다고 말하기 어렵습니다.
    두 모델을 같은 이미지로 짝지어 비교하므로, 따로 구한 신뢰구간을 겹쳐 보는 것보다 정확합니다.
    """
    import random

    a, b = read_counts(run_a), read_counts(run_b)
    stems = sorted(set(a) & set(b))
    lines = [f"| 범위 | 클래스 | 이미지 | MAE {run_a} | MAE {run_b} | 차이 (B−A) [95% CI] |", "|---|---|---|---|---|---|"]
    rng = random.Random(seed)
    for subset in subsets:
        chosen = [s for s in stems if subset == "all" or a[s]["source"] in ("BCCD", "BCDD")]
        n = len(chosen)
        if not n:
            continue
        for cls in ("RBC", "WBC", "Platelets"):
            ea = [abs(int(a[s][f"pred_{cls}"]) - int(a[s][f"gt_{cls}"])) for s in chosen]
            eb = [abs(int(b[s][f"pred_{cls}"]) - int(b[s][f"gt_{cls}"])) for s in chosen]
            diffs = [y - x for x, y in zip(ea, eb)]
            boots = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
            lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
            lines.append(f"| {subset} | {cls} | {n} | {sum(ea) / n:.2f} | {sum(eb) / n:.2f} | "
                         f"{sum(diffs) / n:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 결과 비교 표")
    parser.add_argument("--only", nargs="*", default=None, help="이 접두어로 시작하는 실험만")
    parser.add_argument("--paired", nargs=2, metavar=("A", "B"), default=None,
                        help="두 평가 결과를 같은 이미지에서 짝지어 비교")
    args = parser.parse_args()
    if args.paired:
        print(paired_diff(*args.paired))
        return
    groups = load_runs()
    if not groups:
        raise SystemExit("runs/eval/ 에 평가 결과가 없습니다. 먼저 `python -m bloodcount.evaluate`를 실행하세요.")
    table = build_table(groups, args.only)
    (EVAL_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
