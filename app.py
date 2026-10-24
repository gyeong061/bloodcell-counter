"""
app.py — 혈구 카운팅 웹 데모 (Streamlit)

실행:  streamlit run app.py
가중치: configs/predict.yaml 의 weights (환경 변수 BLOODCOUNT_WEIGHTS 로 덮어쓰기 가능)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from bloodcount.common import ROOT
from bloodcount.counting import CLASSES, aggregate_counts
from bloodcount.inference import PREDICT_CONFIG, CellCounter, draw_detections
from bloodcount.report import DISCLAIMER, build_pdf

SAMPLE_DIR = ROOT / "assets" / "samples"
LEGEND = "🟥 RBC(적혈구) · 🟦 WBC(백혈구) · 🟨 Platelets(혈소판)"

st.set_page_config(page_title="혈구 카운터", page_icon="🩸", layout="wide")


@st.cache_resource
def load_counter() -> CellCounter:
    """모델은 앱 기동 시 한 번만 불러옵니다."""
    return CellCounter(PREDICT_CONFIG, weights=os.environ.get("BLOODCOUNT_WEIGHTS"))


try:
    counter = load_counter()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("설정")
    sample_id = st.text_input("샘플 ID (리포트 표시용)", value="")
    manual = st.toggle("신뢰도 임계값 직접 조절", value=False,
                       help="끄면 평가 단계에서 찾은 클래스별 최적 임계값(또는 기본값)을 사용합니다.")
    conf = st.slider("신뢰도 임계값", 0.05, 0.9, counter.conf, 0.05, disabled=not manual)
    hidden = set(st.multiselect("이미지에서 숨길 클래스", CLASSES, default=[],
                                help="RBC가 빽빽할 때 숨기면 WBC·혈소판이 잘 보입니다."))
    st.divider()
    st.caption(f"모델 `{counter.weights.name}` · device `{counter.device}`")
    st.caption(DISCLAIMER)

# ---------------- 본문 ----------------
st.title("🩸 혈구 자동 카운팅")
st.write("혈액 도말 현미경 이미지(시야)를 올리면 **적혈구·백혈구·혈소판**을 찾아 세고, "
         "여러 시야를 합친 검사 보조 리포트를 만듭니다.")

uploads = st.file_uploader("시야 이미지 업로드 (여러 장 가능)", type=["jpg", "jpeg", "png", "bmp", "tif"],
                           accept_multiple_files=True)
samples = sorted(SAMPLE_DIR.glob("*.jpg")) if SAMPLE_DIR.exists() else []
use_samples = bool(samples) and not uploads and st.button("샘플 이미지로 체험하기")
if use_samples:
    st.session_state["use_samples"] = True
if uploads:
    st.session_state["use_samples"] = False

inputs: list[tuple[str, Image.Image]] = []
if uploads:
    inputs = [(f.name, Image.open(f).convert("RGB")) for f in uploads]
elif st.session_state.get("use_samples"):
    inputs = [(p.name, Image.open(p).convert("RGB")) for p in samples]

if not inputs:
    st.info("이미지를 올리거나 샘플로 체험해 보세요.")
    st.stop()

with st.spinner(f"{len(inputs)}개 시야 분석 중..."):
    results = [counter.analyze(img, source=name, conf=conf if manual else None) for name, img in inputs]
    annotated = [draw_detections(img, r.detections) for (_, img), r in zip(inputs, results)]

# 전체 요약
agg = aggregate_counts([r.counts for r in results])
st.subheader(f"전체 요약 · 시야 {agg['fields']}장")
cols = st.columns(5)
for col, name in zip(cols, CLASSES):
    col.metric(name, agg["totals"][name], help=f"시야당 평균 {agg['mean_per_field'][name]}")
ratios = agg["ratios"]
cols[3].metric("WBC / RBC 100", ratios["WBC_per_100_RBC"] if ratios["WBC_per_100_RBC"] is not None else "-")
cols[4].metric("혈소판 / RBC 100",
               ratios["Platelets_per_100_RBC"] if ratios["Platelets_per_100_RBC"] is not None else "-")

flag_rows = [(r.source, f) for r in results for f in r.flags]
if flag_rows:
    with st.expander(f"⚠️ 검토 필요 {len(flag_rows)}건", expanded=True):
        for source, flag in flag_rows:
            st.write(f"- **{source}** — {flag}")

table = pd.DataFrame([{"시야": r.source, **r.counts, "저신뢰 비율": f"{r.low_conf_ratio:.0%}"}
                      for r in results])
st.dataframe(table, hide_index=True, use_container_width=True)

# 다운로드
pdf_bytes = build_pdf(results, annotated, sample_id=sample_id, model_name=counter.weights.name)
json_bytes = json.dumps({"summary": agg, "fields": [r.to_dict() for r in results]},
                        ensure_ascii=False, indent=2).encode("utf-8")
d1, d2 = st.columns(2)
d1.download_button("📄 PDF 리포트 다운로드", pdf_bytes, file_name=f"bloodcount_{sample_id or 'report'}.pdf",
                   mime="application/pdf", use_container_width=True)
d2.download_button("🧾 JSON 다운로드", json_bytes, file_name="bloodcount_results.json",
                   mime="application/json", use_container_width=True)

# 시야별 상세
st.subheader("시야별 결과")
st.caption(LEGEND)
for (name, image), result, drawn in zip(inputs, results, annotated):
    with st.container(border=True):
        st.markdown(f"**{name}** — " + " · ".join(f"{c} {result.counts[c]}" for c in CLASSES))
        for flag in result.flags:
            st.caption(f"⚠️ {flag}")
        left, right = st.columns(2)
        left.image(image, caption="원본", use_container_width=True)
        shown = draw_detections(image, result.detections, hidden=hidden) if hidden else drawn
        right.image(shown, caption="탐지 결과", use_container_width=True)
