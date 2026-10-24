# Hugging Face Spaces(Docker) · 로컬 배포용 — Streamlit 데모
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU 전용 torch를 먼저 설치해 이미지 크기를 줄임
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
# 가중치는 configs/predict.yaml 의 weights 사용 (BLOODCOUNT_WEIGHTS 환경 변수로 덮어쓰기 가능)
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
