import yaml

from bloodcount.compare_runs import build_table, load_runs


def summary(map50, rbc_mae):
    errs = {c: {"mae": rbc_mae if c == "RBC" else 0.5, "bias": 0, "within_tol": 0.5, "mape": None}
            for c in ("RBC", "WBC", "Platelets")}
    ci = {c: {"mae": [0.1, 0.9], "within_tol": [0.3, 0.7]} for c in ("RBC", "WBC", "Platelets")}
    return {"test_images": 10,
            "detection_test": {"all": {"map50": map50, "map50_95": 0.5}},
            "counting_test": {"all": {"tuned_conf": errs, "tuned_conf_ci95": ci}}}


def test_groups_seeds_and_formats(tmp_path):
    for name, s in [("exp_s1", summary(0.90, 2.0)), ("exp_s2", summary(0.92, 3.0)), ("single", summary(0.8, 1.0))]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "eval_summary.yaml").write_text(yaml.safe_dump(s))
    groups = load_runs(tmp_path)
    assert sorted(groups) == ["exp", "single"]
    assert len(groups["exp"]) == 2
    table = build_table(groups)
    assert "0.910 ± 0.014" in table      # 시드 2개 → 평균 ± 표준편차
    assert "1.00 [0.10–0.90]" in table   # 시드 1개 → 신뢰구간


def test_paired_diff(tmp_path, monkeypatch):
    import bloodcount.compare_runs as cr

    monkeypatch.setattr(cr, "EVAL_DIR", tmp_path)
    header = "stem,source,gt_RBC,pred_RBC,gt_WBC,pred_WBC,gt_Platelets,pred_Platelets\n"
    rows_a = "".join(f"i{i},BCCD,10,{12 + i % 2},1,1,0,0\n" for i in range(10))   # 오차 2~3
    rows_b = "".join(f"i{i},BCCD,10,10,1,1,0,0\n" for i in range(10))             # 오차 0
    for run, rows in (("a", rows_a), ("b", rows_b)):
        (tmp_path / run).mkdir()
        (tmp_path / run / "test_counts.csv").write_text(header + rows)
    table = cr.paired_diff("a", "b", n_boot=200)
    assert "| all | RBC | 10 | 2.50 | 0.00 | -2.50" in table
    assert "| smear | RBC | 10 |" in table
