"""
backend/main.py — 혈구 카운팅 REST API (FastAPI)

실행:  uvicorn backend.main:app --host 0.0.0.0 --port 8000
문서:  http://localhost:8000/docs

원 프로젝트 대비 개선
  - 추론을 스레드풀에서 실행 → 추론 중에도 다른 요청(health 등)이 막히지 않음
  - 업로드 크기·개수 제한, 손상된 이미지 400 처리
  - 임시 파일 없이 메모리에서 바로 이미지 디코딩
  - 여러 시야 → PDF 리포트 엔드포인트 추가
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from bloodcount.counting import aggregate_counts

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_FILES = 20


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 기동 시 모델을 한 번만 불러옵니다. (테스트에서는 가짜 카운터를 미리 넣어 둠)"""
    if getattr(app.state, "counter", None) is None:
        from bloodcount.inference import PREDICT_CONFIG, CellCounter

        app.state.counter = CellCounter(PREDICT_CONFIG, weights=os.environ.get("BLOODCOUNT_WEIGHTS"))
    yield


app = FastAPI(title="Blood Cell Counter API", version="0.1.0",
              description="혈액 도말 이미지의 RBC·WBC·혈소판 탐지 및 카운팅 (검사 보조용, 진단 아님)",
              lifespan=lifespan)


def get_counter(request: Request) -> Any:
    counter = getattr(request.app.state, "counter", None)
    if counter is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")
    return counter


async def read_image(file: UploadFile) -> Image.Image:
    """업로드 파일을 검증하고 PIL 이미지로 엽니다."""
    content = await file.read(MAX_FILE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail=f"{file.filename}: 빈 파일입니다.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"{file.filename}: 10MB를 넘습니다.")
    try:
        image = Image.open(io.BytesIO(content))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"{file.filename}: 이미지로 읽을 수 없습니다.") from exc
    return image.convert("RGB")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/model")
def model_info(request: Request) -> dict[str, Any]:
    return get_counter(request).info()


@app.post("/api/v1/count")
async def count(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    """시야 1장 → 클래스별 개수·검토 표시·박스 목록."""
    counter = get_counter(request)
    image = await read_image(file)
    result = await run_in_threadpool(counter.analyze, image, file.filename or "upload")
    return result.to_dict()


@app.post("/api/v1/report", response_class=Response)
async def report(request: Request, files: list[UploadFile] = File(...), sample_id: str = "") -> Response:
    """여러 시야 → 집계 PDF 리포트."""
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"한 번에 최대 {MAX_FILES}장까지 가능합니다.")
    counter = get_counter(request)
    images = [(f.filename or f"field_{i}", await read_image(f)) for i, f in enumerate(files)]

    def run() -> bytes:
        from bloodcount.inference import draw_detections
        from bloodcount.report import build_pdf

        results = [counter.analyze(img, source=name) for name, img in images]
        drawn = [draw_detections(img, r.detections) for (_, img), r in zip(images, results)]
        return build_pdf(results, drawn, sample_id=sample_id, model_name=counter.weights.name)

    pdf = await run_in_threadpool(run)
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="bloodcount_report.pdf"'})


@app.post("/api/v1/summary")
async def summary(request: Request, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    """여러 시야 → 집계 JSON (PDF 없이 숫자만 필요할 때)."""
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"한 번에 최대 {MAX_FILES}장까지 가능합니다.")
    counter = get_counter(request)
    results = []
    for f in files:
        image = await read_image(f)
        results.append(await run_in_threadpool(counter.analyze, image, f.filename or "upload"))
    return {"summary": aggregate_counts([r.counts for r in results]),
            "fields": [{"source": r.source, "counts": r.counts, "flags": r.flags} for r in results]}
