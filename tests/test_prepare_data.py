from bloodcount.prepare_data import Box, ParseReport, parse_voc, to_yolo_line

XML = """<annotation><size><width>640</width><height>480</height></size>
<object><name>WBC</name><bndbox><xmin>260</xmin><ymin>177</ymin><xmax>491</xmax><ymax>376</ymax></bndbox></object>
<object><name>RBC</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>40</xmax><ymax>500</ymax></bndbox></object>
<object><name>RBC</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>10</xmax><ymax>40</ymax></bndbox></object>
<object><name>Neutrophil</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object>
</annotation>"""


def test_parse_voc_fixes_and_drops(tmp_path):
    path = tmp_path / "BloodImage_99999.xml"
    path.write_text(XML)
    report = ParseReport()
    width, height, boxes = parse_voc(path, report)

    assert (width, height) == (640, 480)
    assert len(boxes) == 2                       # 면적 0 박스·모르는 클래스 제외
    assert boxes[1].xmin == 40 and boxes[1].xmax == 100   # 뒤집힌 x 정렬
    assert boxes[1].ymax == 480                  # 이미지 밖 y 클리핑
    assert len(report.fixed) == 1
    assert len(report.dropped) == 1
    assert report.unknown_class["Neutrophil"] == 1


def test_to_yolo_line_normalizes():
    line = to_yolo_line(Box(1, 0, 0, 320, 240), 640, 480)
    assert line == "1 0.250000 0.250000 0.500000 0.500000"


def test_remap_txl_label_reorders_classes_and_validates():
    from bloodcount.prepare_data import remap_txl_label

    report = ParseReport()
    text = "\n".join([
        "0 0.5 0.5 0.2 0.2",   # TXL WBC → 우리 1
        "1 0.3 0.3 0.1 0.1",   # TXL RBC → 우리 0
        "2 0.1 0.1 0.05 0.05",  # Platelets → 2
        "1 1.5 0.3 0.1 0.1",   # 좌표 범위 밖 → 제외
        "1 0.3 0.3 0.1",       # 값 4개 → 제외
        "7 0.3 0.3 0.1 0.1",   # 모르는 클래스
    ])
    lines = remap_txl_label(text, "img", report)
    assert [line.split()[0] for line in lines] == ["1", "0", "2"]
    assert len(report.dropped) == 2
    assert report.unknown_class["7"] == 1
