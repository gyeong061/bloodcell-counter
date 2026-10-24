"""
bloodcount/export_web.py — 학습한 모델을 브라우저 데모(web/)용으로 내보내기

브라우저 데모(web/index.html)는 서버 없이 onnxruntime-web으로 모델을 직접 돌립니다.
이 스크립트는 그 데모가 읽는 파일을 만듭니다.

  web/model.partN.txt  ONNX 모델(입력 크기 가변)을 base64로 나눈 조각
                       (일부 호스팅이 .onnx 파일을 서빙하지 않아 텍스트로 보관)
  web/config.json      클래스별 임계값·NMS 설정·예시 이미지 목록·모델 설명
  web/samples/         예시 이미지 (assets/samples/에서 복사)

[사용 예]
  python -m bloodcount.export_web                                   # configs/predict.yaml 기준
  python -m bloodcount.export_web --weights runs/detect/txl_yolo11s/weights/best.pt \\
         --from-eval runs/eval/txl_yolo11s/eval_summary.yaml          # 평가에서 찾은 임계값 사용
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import tempfile
from pathlib import Path

from bloodcount.common import CONFIG_DIR, ROOT, load_yaml, resolve_path

WEB_DIR = ROOT / "web"
SAMPLE_DIR = ROOT / "assets" / "samples"
PART_BYTES = 6 * 1024 * 1024  # 조각당 원본 6MB → base64 약 8MB


def export_onnx(weights: Path, imgsz: int) -> Path:
    """가중치를 임시 폴더로 복사해 ONNX로 내보냅니다 (원래 폴더를 건드리지 않음)."""
    from ultralytics import YOLO

    tmp = Path(tempfile.mkdtemp())
    copy = tmp / weights.name
    shutil.copy2(weights, copy)
    return Path(YOLO(str(copy)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=True))


def split_base64(onnx_path: Path, out_dir: Path) -> list[str]:
    """ONNX 파일을 PART_BYTES 단위로 잘라 base64 텍스트 조각으로 저장합니다."""
    for old in out_dir.glob("model.part*.txt"):
        old.unlink()
    data = onnx_path.read_bytes()
    names = []
    for i in range(0, len(data), PART_BYTES):
        name = f"model.part{i // PART_BYTES + 1}.txt"
        (out_dir / name).write_text(base64.b64encode(data[i:i + PART_BYTES]).decode())
        names.append(name)
    return names


def describe_model(weights: Path) -> str:
    """학습 run의 args.yaml로 데모에 표시할 한 줄 설명을 만듭니다."""
    args_file = weights.parent.parent / "args.yaml"
    if not args_file.exists():
        return f"{weights.parent.parent.name} 가중치."
    args = load_yaml(args_file)
    model = Path(str(args.get("model", "yolo"))).stem.replace("yolo", "YOLO")
    data = str(args.get("data", ""))
    data_name = ("TXL-PBC 재라벨 BCCD 시야" if "ablation" in data and "txl" in data else
                 "BCCD 원본 라벨" if "bccd" in data else
                 "TXL-PBC" if "txl" in data else Path(data).parent.name)
    return f"{model} · {data_name}로 학습 · {args.get('epochs', '?')} epoch."


def main() -> None:
    parser = argparse.ArgumentParser(description="브라우저 데모용 모델·설정 내보내기")
    parser.add_argument("--weights", type=Path, default=None, help="기본: configs/predict.yaml의 weights")
    parser.add_argument("--from-eval", type=Path, default=None,
                        help="이 eval_summary.yaml의 tuned_conf_from_val을 임계값으로 사용")
    parser.add_argument("--out", type=Path, default=WEB_DIR)
    args = parser.parse_args()

    cfg = load_yaml(CONFIG_DIR / "predict.yaml")
    params = cfg.get("parameters", {})
    weights = resolve_path(args.weights or cfg["weights"], must_exist=True)
    class_conf = dict(cfg.get("class_conf") or {})
    if args.from_eval:
        class_conf = load_yaml(resolve_path(args.from_eval, must_exist=True))["tuned_conf_from_val"]
    if not class_conf:
        conf = float(params.get("conf", 0.25))
        class_conf = {"RBC": conf, "WBC": conf, "Platelets": conf}

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    imgsz = int(params.get("imgsz", 640))
    print(f"ONNX 내보내기: {weights}")
    parts = split_base64(export_onnx(weights, imgsz), out)

    samples = json.loads((SAMPLE_DIR / "samples.json").read_text(encoding="utf-8"))
    shutil.rmtree(out / "samples", ignore_errors=True)
    (out / "samples").mkdir()
    for sample in samples:
        shutil.copy2(SAMPLE_DIR / Path(sample["file"]).name, out / "samples" / Path(sample["file"]).name)

    config = {
        "model_note": describe_model(weights),
        "weights": weights.parent.parent.name,
        "class_conf": {k: float(v) for k, v in class_conf.items()},
        "iou": float(params.get("iou", 0.5)),
        "imgsz": imgsz,
        "max_det": int(params.get("max_det", 300)),
        "model_parts": parts,
        "samples": samples,
    }
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in config.items() if k != "samples"}, ensure_ascii=False, indent=1))
    print(f"완료 → {out}  (로컬 확인: python -m http.server -d web 8000)")


if __name__ == "__main__":
    main()
