import pytest

from bloodcount.counting import (Detection, aggregate_counts, cell_ratios, count_cells,
                                 count_errors, review_flags)


def det(name, conf=0.9):
    return Detection(name, conf, (0.0, 0.0, 10.0, 10.0))


def test_count_cells_fills_missing_classes_with_zero():
    counts = count_cells([det("RBC"), det("RBC"), det("WBC"), det("Unknown")])
    assert counts == {"RBC": 2, "WBC": 1, "Platelets": 0}


def test_aggregate_counts_totals_and_mean():
    agg = aggregate_counts([{"RBC": 10, "WBC": 1, "Platelets": 0},
                            {"RBC": 20, "WBC": 1, "Platelets": 2}])
    assert agg["fields"] == 2
    assert agg["totals"] == {"RBC": 30, "WBC": 2, "Platelets": 2}
    assert agg["mean_per_field"]["RBC"] == 15
    assert agg["ratios"]["WBC_per_100_RBC"] == pytest.approx(6.67, abs=0.01)


def test_ratios_none_when_no_rbc():
    assert cell_ratios({"RBC": 0, "WBC": 3}) == {"WBC_per_100_RBC": None, "Platelets_per_100_RBC": None}


def test_review_flags():
    assert review_flags({"RBC": 12, "WBC": 1, "Platelets": 1}) == []
    flags = review_flags({"RBC": 2, "WBC": 0, "Platelets": 0}, low_conf_ratio=0.5)
    assert len(flags) == 3


def test_count_errors_metrics():
    gt = [{"RBC": 10, "WBC": 1, "Platelets": 0}, {"RBC": 10, "WBC": 1, "Platelets": 2}]
    pred = [{"RBC": 8, "WBC": 1, "Platelets": 0}, {"RBC": 11, "WBC": 1, "Platelets": 2}]
    err = count_errors(pred, gt)
    assert err["RBC"]["mae"] == 1.5
    assert err["RBC"]["bias"] == -0.5
    assert err["RBC"]["within_tol"] == 0.5
    assert err["WBC"]["mae"] == 0
    assert err["Platelets"]["mape"] == 0  # 정답 0인 이미지는 상대 오차에서 제외


def test_count_errors_length_mismatch():
    with pytest.raises(ValueError):
        count_errors([{}], [])


def test_bootstrap_ci_brackets_point_estimate():
    from bloodcount.counting import bootstrap_ci

    gt = [{"RBC": 10, "WBC": 1, "Platelets": 0}] * 20
    pred = [{"RBC": 10 + (i % 4), "WBC": 1, "Platelets": 0} for i in range(20)]  # 오차 0,1,2,3 반복
    ci = bootstrap_ci(pred, gt, n_boot=500)
    lo, hi = ci["RBC"]["mae"]
    assert lo <= 1.5 <= hi            # 실제 MAE 1.5
    assert ci["WBC"]["mae"] == [0.0, 0.0]
    assert 0 <= ci["RBC"]["within_tol"][0] <= 0.5 <= ci["RBC"]["within_tol"][1] <= 1
