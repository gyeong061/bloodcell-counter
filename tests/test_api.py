import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from bloodcount.counting import Detection, count_cells, review_flags
from bloodcount.inference import FieldResult


class FakeCounter:
    """ultralytics 없이 API 동작만 검증하기 위한 가짜 카운터."""

    class weights:  # noqa: N801 — CellCounter.weights.name 흉내
        name = "fake.pt"

    def info(self):
        return {"weights": "fake.pt"}

    def analyze(self, image, source="upload", conf=None):
        dets = [Detection("RBC", 0.9, (1, 1, 20, 20)), Detection("WBC", 0.8, (30, 30, 60, 60))]
        counts = count_cells(dets)
        return FieldResult(source, image.width, image.height, dets, counts, 0.0, review_flags(counts))


def jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buf, format="JPEG")
    return buf.getvalue()


app.state.counter = FakeCounter()
client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_count_returns_counts():
    res = client.post("/api/v1/count", files={"file": ("a.jpg", jpeg_bytes(), "image/jpeg")})
    assert res.status_code == 200
    body = res.json()
    assert body["counts"] == {"RBC": 1, "WBC": 1, "Platelets": 0}
    assert body["image_width"] == 64


def test_count_rejects_non_image():
    res = client.post("/api/v1/count", files={"file": ("a.jpg", b"not an image", "image/jpeg")})
    assert res.status_code == 400


def test_summary_aggregates_fields():
    files = [("files", (f"{i}.jpg", jpeg_bytes(), "image/jpeg")) for i in range(3)]
    body = client.post("/api/v1/summary", files=files).json()
    assert body["summary"]["totals"]["RBC"] == 3


def test_report_returns_pdf():
    files = [("files", ("a.jpg", jpeg_bytes(), "image/jpeg"))]
    res = client.post("/api/v1/report", files=files)
    assert res.status_code == 200
    assert res.content[:4] == b"%PDF"
