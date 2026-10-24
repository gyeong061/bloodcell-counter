"""
bloodcount/report.py — 여러 시야 분석 결과를 PDF 검사 보조 리포트로 만듭니다.

구성: 제목·면책 문구 → 전체 집계표 → 비율 → 검토 필요 항목 → 시야별 표 → 시야별 이미지
한글은 reportlab 내장 CID 폰트(HYGothic-Medium)로 출력해 별도 폰트 파일이 필요 없습니다.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Sequence

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from bloodcount.counting import CLASSES, aggregate_counts
from bloodcount.inference import FieldResult

FONT = "HYGothic-Medium"
pdfmetrics.registerFont(UnicodeCIDFont(FONT))

STYLE_TITLE = ParagraphStyle("title", fontName=FONT, fontSize=18, leading=24, spaceAfter=6)
STYLE_H2 = ParagraphStyle("h2", fontName=FONT, fontSize=12.5, leading=18, spaceBefore=10, spaceAfter=4)
STYLE_BODY = ParagraphStyle("body", fontName=FONT, fontSize=9.5, leading=14)
STYLE_NOTE = ParagraphStyle("note", fontName=FONT, fontSize=8.5, leading=12,
                            textColor=colors.HexColor("#555555"))

DISCLAIMER = ("본 리포트는 공개 연구 데이터(BCCD·TXL-PBC)로 만든 AI 모델의 자동 집계 결과이며, "
              "의학적 진단이나 임상 검사 결과를 대신하지 않습니다. 모든 수치는 전문가가 직접 확인해야 합니다.")


def _table(rows: list[list], col_widths: list[float] | None = None) -> Table:
    table = Table(rows, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6F9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return table


def _pil_to_flowable(image: Image.Image, max_width: float) -> RLImage:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    buf.seek(0)
    ratio = image.height / image.width
    return RLImage(buf, width=max_width, height=max_width * ratio)


def build_pdf(results: Sequence[FieldResult], annotated: Sequence[Image.Image] | None = None,
              sample_id: str = "", model_name: str = "") -> bytes:
    """분석 결과로 PDF 바이트를 만듭니다. annotated는 results와 같은 순서의 박스 그린 이미지."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="혈구 카운팅 검사 보조 리포트")
    width = doc.width
    agg = aggregate_counts([r.counts for r in results])
    story: list = [
        Paragraph("혈구 카운팅 검사 보조 리포트", STYLE_TITLE),
        Paragraph(f"생성 시각: {datetime.now():%Y-%m-%d %H:%M} · 샘플 ID: {sample_id or '-'} · "
                  f"모델: {model_name or '-'} · 분석 시야: {agg['fields']}장", STYLE_BODY),
        Spacer(1, 4),
        Paragraph(DISCLAIMER, STYLE_NOTE),
        Paragraph("1. 전체 집계", STYLE_H2),
        _table([["구분", *CLASSES],
                ["합계", *[agg["totals"][c] for c in CLASSES]],
                ["시야당 평균", *[agg["mean_per_field"][c] for c in CLASSES]]],
               [width * 0.31] + [width * 0.23] * 3),
    ]

    ratios = agg["ratios"]
    ratio_text = ("RBC가 검출되지 않아 비율을 계산하지 않았습니다." if ratios["WBC_per_100_RBC"] is None else
                  f"RBC 100개당 WBC {ratios['WBC_per_100_RBC']}개 · 혈소판 {ratios['Platelets_per_100_RBC']}개 "
                  "(촬영한 시야 기준 상대 비율이며 혈중 농도가 아닙니다)")
    story += [Paragraph("2. 상대 비율", STYLE_H2), Paragraph(ratio_text, STYLE_BODY)]

    flagged = [(r.source, f) for r in results for f in r.flags]
    story.append(Paragraph("3. 검토 필요 항목", STYLE_H2))
    if flagged:
        story.append(_table([["시야", "내용"]] + [[s, Paragraph(f, STYLE_BODY)] for s, f in flagged],
                            [width * 0.3, width * 0.7]))
    else:
        story.append(Paragraph("자동 점검에서 특이 사항이 없었습니다.", STYLE_BODY))

    story.append(Paragraph("4. 시야별 결과", STYLE_H2))
    rows = [["시야", *CLASSES, "저신뢰 비율"]]
    rows += [[r.source, *[r.counts[c] for c in CLASSES], f"{r.low_conf_ratio:.0%}"] for r in results]
    story.append(_table(rows, [width * 0.36] + [width * 0.16] * 4))

    if annotated:
        story += [PageBreak(), Paragraph("5. 시야별 탐지 이미지", STYLE_H2),
                  Paragraph("빨강 RBC · 파랑 WBC · 노랑 혈소판", STYLE_NOTE)]
        for result, image in zip(results, annotated):
            story += [Spacer(1, 6), Paragraph(f"{result.source} — "
                      + " · ".join(f"{c} {result.counts[c]}" for c in CLASSES), STYLE_BODY),
                      _pil_to_flowable(image, width * 0.8)]

    doc.build(story)
    return buf.getvalue()
