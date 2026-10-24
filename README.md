# 🩸 Blood Cell Counter — 혈액 도말 이미지 혈구 자동 카운팅

혈액 도말 현미경 이미지를 올리면 **적혈구(RBC)·백혈구(WBC)·혈소판(Platelets)** 을 YOLO11로 탐지해 세고,
여러 시야를 합친 **검사 보조 PDF 리포트**를 만들어 주는 서비스입니다.

> ⚠️ 공개 연구 데이터로 만든 포트폴리오 프로젝트입니다. 의학적 진단을 대신하지 않습니다.

![앱 화면](docs/app_screenshot.png)

## 한눈에 보기

1. **문제 정의** — "박스를 잘 치는 모델"이 아니라 "개수를 정확히 세는 서비스". mAP와 함께 이미지별 개수 오차를 직접 설계해 평가
2. **데이터를 의심하기** — BCCD로 시작했다가 RBC 개수가 이상하게 맞는 현상을 추적 → 정답 라벨의 RBC 누락 + 오류 상쇄를 발견
3. **원인 해결과 검증** — 다시 라벨링된 TXL-PBC로 교체하고, **이미지는 같고 라벨만 다른 통제 실험**으로 라벨 품질의 효과를 숫자로 확인
4. **끝까지 만들기** — 학습 → 평가 → REST API → 웹 데모 → PDF 리포트 → Docker

## 주요 기능

| 기능 | 설명 |
|---|---|
| 혈구 탐지·카운팅 | 시야별 RBC·WBC·혈소판 개수, 클래스별 색 박스 |
| 여러 시야 집계 | 합계, 시야당 평균, RBC 100개당 WBC·혈소판 비율 |
| 검토 필요 표시 | RBC 과소 검출, WBC 미검출, 저신뢰 검출 비율이 높은 시야를 알려줌 (임상 판단 아님) |
| PDF 리포트 | 집계표·검토 항목·시야별 탐지 이미지를 한 파일로 |
| REST API | `/api/v1/count`, `/api/v1/summary`, `/api/v1/report` (FastAPI) |
| 브라우저 데모 | 서버 없이 브라우저 안에서 모델 실행(onnxruntime-web). 사진이 밖으로 전송되지 않음 → GitHub Pages로 공개 |

## 빠른 시작

```bash
pip install -r requirements-dev.txt
python -m bloodcount.prepare_data          # BCCD·TXL-PBC 다운로드 → YOLO 형식 통일
python -m bloodcount.eda --dataset bccd    # BCCD 탐색
python -m bloodcount.eda                   # TXL-PBC 탐색 + 라벨 비교 + 누수 의심 목록 생성
python -m bloodcount.train                 # 본학습 (GPU 권장 → notebooks/colab_train.ipynb)
python -m bloodcount.evaluate              # Test 평가 (mAP + 카운트 오차 + 95% 신뢰구간, 출처별)
python -m bloodcount.compare_runs          # 실험 비교 표 (시드 평균 ± 표준편차)
python -m bloodcount.export_web            # 브라우저 데모용 모델·설정 → web/
streamlit run app.py                       # 웹 앱 (PDF 리포트 포함)
uvicorn backend.main:app --port 8000       # API (http://localhost:8000/docs)
pytest -q                                  # 테스트
```

GPU가 없으면 [`notebooks/colab_train.ipynb`](notebooks/colab_train.ipynb)를 Colab(T4)에서 실행하세요. 실험 1·2와 결과 표, 웹 데모 내보내기까지 한 번에 진행됩니다.

**브라우저 데모 공개**: 저장소 Settings → Pages → Source를 "GitHub Actions"로 바꾸면 `web/` 폴더가 자동 배포됩니다 (`.github/workflows/pages.yml`).
로컬 확인은 `python -m http.server -d web 8000` 후 http://localhost:8000.

## 1. 데이터를 바꾸게 된 과정

### BCCD로 시작

[BCCD](https://github.com/Shenggan/BCCD_Dataset) (364장, MIT)로 첫 모델을 만들었더니 WBC·혈소판은 잘 셌지만,
RBC는 기본 임계값에서 **이미지당 약 7개를 더 세는** 것으로 나왔습니다. 탐지 결과를 직접 보니 대부분 실제로 있는 RBC였습니다.

![라벨 누락 예시](docs/label_missing_example.jpg)
*BCCD test/BloodImage_00137 — 초록: 정답 라벨 RBC 5개 · 빨강: 정답과 겹치지 않는 모델 검출 13개*

임계값을 0.6으로 올리면 RBC 개수 오차가 2.9개로 줄었지만, 이때 모델은 **라벨에 있는 RBC의 27%를 놓치면서
라벨에 없는 RBC를 158개 더 세고 있었습니다.** 두 오류가 상쇄돼 합계만 맞아 보인 것입니다.
→ 지표가 좋아지는 것과 실제로 정확해지는 것은 다를 수 있습니다.

### TXL-PBC로 교체

[TXL-PBC](https://github.com/lugan113/TXL-PBC_Dataset) (Gan et al., *Scientific Data* 12, 1694, 2025, MIT)는
BCCD를 포함한 공개 데이터 4개를 모아 사람이 다시 라벨링한 데이터셋입니다. 같은 BCCD 이미지 160장을 비교하면:

| 클래스 | BCCD 원본 라벨 | TXL-PBC 재라벨 | 변화 |
|---|---:|---:|---:|
| RBC | 1,876 | 3,179 | **+69%** (160장 중 159장에서 추가) |
| WBC | 163 | 164 | 거의 같음 |
| Platelets | 131 | 167 | +27% |

![라벨 비교](runs/eda/txl/relabel_comparison.png)

### TXL-PBC 구성 (1,260장, 공식 분할 train 882 / val 252 / test 126)

| 출처 | 이미지 | 크기 | 이미지당 RBC | 특징 |
|---|---:|---|---:|---|
| BCCD | 160 | 640×480 | 19.9 | 넓은 시야 도말 |
| BCDD | 100 | 256×256 | 16.3 | 넓은 시야, 혈소판 거의 없음 |
| PBC | 500 | 360×363 | 12.6 | WBC 중심으로 자른 이미지 |
| Raabin-WBC | 500 | 575×575 | 10.3 | WBC 중심, 다른 장비·염색 |

- 원본 클래스 순서(WBC=0, RBC=1)를 이름 기준으로 RBC=0, WBC=1, Platelets=2로 다시 매핑
- `.png` 확장자지만 실제 내용은 JPEG인 파일을 재인코딩 없이 `.jpg`로 저장
- 테스트 이미지 1장에서 중복 라벨 13개 발견 (평가 시 Ultralytics가 자동 제거) — 재라벨 데이터도 완벽하진 않음

### 누수 점검 (이전 팀 프로젝트에서 배운 점)

| 점검 | 방법 | 결과 |
|---|---|---|
| 분할 간 거의 같은 이미지 | 지각 해시(aHash) 거리 ≤ 30 | BCCD 1쌍, TXL-PBC 2쌍 (눈으로 같은 세포·시야 확인) |
| 같은 촬영에서 나온 이미지 | Raabin-WBC 파일명 `<촬영ID>_<번호>` 그룹 | 11개 촬영이 서로 다른 분할에 나뉘어 있음 |
| 다른 데이터셋과 교차 누수 | metadata의 원본 파일명 비교 | TXL test의 BCCD 18장 중 16장이 BCCD train/val과 같은 사진 |

의심 이미지(val 10장, test 2장)는 `data/txl/suspect_leaks.yaml`로 저장되고 `evaluate.py`가 자동으로 제외합니다.
다른 데이터셋으로 학습한 모델을 평가할 때는 `--exclude-seen-by`로 교차 누수 이미지를 뺍니다.

## 2. 실험 1 — 라벨 품질 통제 실험

**질문**: RBC 라벨 누락은 모델을 얼마나 망가뜨리나?

**설계**: 이미지·모델·설정은 똑같고 라벨만 다른 두 모델을 학습합니다 (`make_label_ablation.py`).

- 이미지: TXL-PBC train/val에 있는 BCCD 이미지 (train 101장, val 41장)
- 라벨 A: BCCD 원본 (학습 RBC 박스 1,186개) / 라벨 B: TXL-PBC 재라벨 (학습 RBC 박스 1,941개)
- 공통: YOLO11n, 30 epoch, seed 42, 같은 증강 / 평가: TXL-PBC test 124장 (누수 의심 2장 제외)

| 지표 (TXL-PBC test) | 원본 라벨로 학습 | 재라벨로 학습 | 차이 |
|---|---:|---:|---:|
| mAP50 | 0.878 | **0.908** | +3.0%p |
| mAP50-95 | 0.637 | **0.663** | +2.6%p |
| **RBC Recall** | 0.832 | **0.957** | **+12.5%p** |
| RBC 개수 MAE (튜닝 임계값) | 2.86 | **2.51** | −12% |
| RBC 개수 ±1 일치율 | 33.9% | **45.2%** | +11.3%p |
| WBC 개수 MAE | 0.52 | 0.47 | |
| 혈소판 개수 MAE | 0.21 | 0.17 | |

**RBC Recall은 4개 출처 모두에서 올랐습니다** (BCCD 0.79→0.94, BCDD 0.80→0.91, PBC 0.88→0.97, Raabin 0.79→0.94).
라벨에서 빠진 RBC를 "배경"으로 배운 모델이 실제 RBC를 놓치는 경향이, 라벨을 고치자 사라진 것입니다.

**개수 오차는 얼마나 확실한 차이인가** — 같은 test 이미지에서 두 모델의 오차를 짝지어 비교했습니다 (부트스트랩 95% 신뢰구간, `compare_runs --paired`).

| 범위 | RBC MAE (원본 → 재라벨) | 차이 [95% CI] |
|---|---:|---:|
| 전체 124장 | 2.85 → 2.51 | −0.35 [−0.65, −0.05] |
| 넓은 시야 도말 32장 (BCCD+BCDD) | 3.53 → 3.47 | −0.06 [−0.72, +0.62] |

전체에서는 개선이 확인되지만 구간이 0에 가깝고, **서비스 대상인 넓은 시야 도말에서는 차이를 말할 수 없습니다.**
도말 test가 32장뿐이고, 두 모델 모두 도말에서 RBC를 이미지당 약 2개 더 세는 경향(편향 +1.8, +2.3)이 남아 있습니다.
→ 라벨 품질의 확실한 효과는 **RBC를 놓치지 않게 된 것(Recall)** 이고, 개수 정확도는 실험 2(더 많은 데이터)와 시드 반복으로 다시 확인합니다.

**주의할 점**

- 재라벨 모델은 기본 임계값(0.25)에서는 RBC를 이미지당 3.8개 더 세어 오히려 MAE가 큽니다(4.47).
  더 적극적으로 찾는 모델이라 **임계값 튜닝이 필수**이고, val에서 고른 0.65를 적용해야 위 결과가 나옵니다.
- 시드 하나로 한 번씩만 학습한 결과라, mAP 3%p 차이는 우연일 가능성도 있습니다 → Colab에서 시드 3개로 반복.
  반면 RBC Recall 향상은 크고 4개 출처에서 일관돼 신뢰할 만합니다.
- BCCD 이미지로만 학습해 Raabin-WBC(다른 장비·염색)에서는 WBC mAP50이 0.54로 낮습니다 → 실험 2에서 전체 TXL-PBC로 학습

## 3. 실험 2 — TXL-PBC 본학습 (Colab에서 진행)

전체 TXL-PBC(882장)로 YOLO11n(Baseline)과 YOLO11s(본학습)를 **같은 100 epoch·patience 20**으로 학습해 모델 크기 효과를 비교합니다.
실험 1도 시드 3개(42·7·2024)로 반복해, 라벨 품질 효과가 시드에 따라 유지되는지 확인합니다.

<!-- EXP2_RESULTS: Colab 노트북 '4. 결과 비교'의 표(runs/eval/comparison.md)를 여기에 붙여 넣으세요 -->

## 설계 포인트

- **도메인 맞춤 증강**: 현미경 이미지는 방향에 의미가 없어 상하·좌우 반전을 허용(풍력 터빈 프로젝트에서는 금지했던 설정).
  염색 차이에 대응해 색조·채도 변화를 주고, 혈구 경계를 흐리는 mixup과 박스를 키우는 회전은 사용하지 않음
- **공정한 비교**: 비교하는 실험끼리 epoch·patience·seed를 같게 맞춤
- **카운팅 지표**: mAP 외에 이미지별 개수 오차(MAE, 편향, ±1 일치율)를 전체·출처별로 보고
- **임계값 튜닝**: val에서 클래스별로 개수 오차가 가장 작은 임계값을 찾아 test와 서비스에 적용 (`configs/predict.yaml`의 `class_conf`)

## 프로젝트 구조

```
bloodcount/
├── common.py              # 공용 함수 (경로·YAML·장치 선택·메타데이터·교차 누수)
├── prepare_data.py        # BCCD(VOC)·TXL-PBC(YOLO) → 클래스 순서 통일, 라벨 검증
├── eda.py                 # 분포·출처별 특성·라벨 비교·누수 의심 목록
├── make_label_ablation.py # 실험 1: 같은 이미지 × 다른 라벨 데이터 생성
├── train.py               # 학습 (OOM 시 batch 자동 축소)
├── evaluate.py            # Test 평가: mAP + 카운트 오차 (전체·출처별) + 임계값 탐색 + 누수 제외
├── counting.py            # 개수·집계·비율·검토 표시·오차 지표 (모델 의존 없음)
├── inference.py           # CellCounter: 추론 엔진 (CLI·앱·API 공용)
├── predict.py             # 폴더 일괄 분석 → JSON·이미지·PDF
├── report.py              # PDF 리포트 생성
├── compare_runs.py        # 실험 비교 표 (시드 평균 ± 표준편차 / 신뢰구간)
└── export_web.py          # 브라우저 데모용 ONNX·설정 내보내기
backend/main.py            # FastAPI
app.py                     # Streamlit 웹 앱
web/                       # 브라우저 데모 (GitHub Pages로 배포)
configs/                   # 학습·추론 설정 (하이퍼파라미터 하드코딩 금지)
notebooks/                 # Colab 학습 노트북 (실험 1·2)
tests/                     # pytest (카운팅·변환·누수 점검·API)
```

## 검토했지만 하지 않은 것: WBC 종류 분류

실제 도말 검토에서 가치가 큰 기능은 백혈구를 종류별로 나누는 감별(호중구·림프구 등)이라 추가를 검토했지만, 데이터를 확인한 뒤 범위에서 뺐습니다.

| 확인한 것 | 결과 | 판단 |
|---|---|---|
| 사용할 수 있는 라벨 | TXL-PBC 안의 PBC 500장은 파일명에 종류가 있음 (BA·BNE·EO·LY·MO 각 100장) | 추가 라벨링 없이 분류기 학습은 가능 |
| 종류 구성 | 혈액에서 가장 흔한 **분엽호중구가 없음** (띠호중구만 있음) | 실제 도말에 쓸 수 없음 |
| 평가 데이터 | test가 종류당 8~12장 | 정확도를 신뢰할 수 없음 |
| 도메인 | PBC는 다른 장비로 백혈구를 가운데 두고 찍은 이미지 | 넓은 시야 도말(서비스 대상)에서 동작한다는 근거 없음 |

원본 PBC 데이터셋(8종류)으로 확장하면 위 제약 대부분이 풀릴 수 있어 다음 단계로 둡니다.

## 한계와 다음 단계

- 현재 앱·데모의 기본 모델은 실험 1의 재라벨 모델(BCCD 이미지 101장, CPU 30 epoch) → 실험 2 모델로 교체 예정
- 모든 이미지가 공개 연구 데이터라, 실제 병원의 현미경·염색·촬영 조건에서 성능이 유지될지는 미검증
- 혈소판 test 박스가 49개뿐이고 넓은 시야 도말 test는 32장이라 해당 지표의 불확실성이 큼 (신뢰구간 참고)
- 추론 입력 방식에 따라 RBC 개수가 달라지는 것을 발견: 640×480 그대로 넣을 때보다 640×640으로 패딩해 넣을 때 정답에 더 가까운 예시가 있었음(예: 정답 20개에 27개 vs 22개).
  학습·평가·서비스의 입력 방식을 맞추는 실험이 필요
- Python 앱과 브라우저 데모가 같은 후처리를 따로 구현 → 예시 6장의 개수가 같은 것은 확인했고, 자동 비교 테스트는 다음 단계

## 출처

- 데이터: BCCD Dataset (MIT) · TXL-PBC (MIT) — Gan, L., Li, X. & Wang, X. *A Curated and Re-annotated Peripheral Blood Cell Dataset Integrating Four Public Resources.* Sci Data 12, 1694 (2025). doi:10.1038/s41597-025-05980-z
- 모델: [Ultralytics YOLO11](https://docs.ultralytics.com/) (AGPL-3.0)
- 파이프라인 구조는 팀 프로젝트 [wind-turbine-yolo](https://github.com/gyeong061/AIFFEL_QUEST_ENG/tree/main/Main_Quest/MQ02)를 바탕으로 개인이 재설계·확장
