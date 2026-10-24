import csv

from bloodcount.common import seen_originals
from bloodcount.eda import capture_group, find_suspect_leaks
from bloodcount.evaluate import select_images


def write_meta(root, rows):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "metadata.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "split", "source", "original_name"])
        w.writeheader()
        w.writerows(rows)


def test_capture_group_groups_raabin_crops():
    assert capture_group("Raabin-WBC", "20190604_110533_0.jpg") == capture_group("Raabin-WBC", "20190604_110533_1.jpg")
    assert capture_group("PBC", "BA_1_0.jpg") != capture_group("PBC", "BA_1_1.jpg")


def test_find_suspect_leaks_uses_groups_and_hash_pairs(tmp_path):
    write_meta(tmp_path, [
        {"stem": "a", "split": "train", "source": "Raabin-WBC", "original_name": "X_0.jpg"},
        {"stem": "b", "split": "test", "source": "Raabin-WBC", "original_name": "X_1.jpg"},
        {"stem": "c", "split": "val", "source": "PBC", "original_name": "P1.jpg"},
        {"stem": "d", "split": "test", "source": "PBC", "original_name": "P2.jpg"},
    ])
    dups = [{"a": "train/zz", "b": "val/c", "distance": 20}]
    suspects = find_suspect_leaks(tmp_path, dups)
    assert suspects == {"val": ["c"], "test": ["b"]}


def test_seen_originals_and_select_images(tmp_path):
    other = tmp_path / "bccd"
    write_meta(other, [
        {"stem": "B1", "split": "train", "source": "BCCD", "original_name": "B1.jpg"},
        {"stem": "B2", "split": "test", "source": "BCCD", "original_name": "B2.jpg"},
    ])
    assert seen_originals(other) == {"B1.jpg"}

    data = tmp_path / "txl"
    (data / "images" / "test").mkdir(parents=True)
    for stem in ("t1", "t2", "t3"):
        (data / "images" / "test" / f"{stem}.jpg").write_bytes(b"")
    meta = {"t1": {"original_name": "B1.jpg"}, "t2": {"original_name": "B2.jpg"},
            "t3": {"original_name": "P.jpg"}}
    kept, dropped = select_images(data, "test", {"t3"}, {"B1.jpg"}, meta)
    assert [p.stem for p in kept] == ["t2"]
    assert dropped == 2
