"""Train a YOLO object detector for the fauna camera tracker.

Expected dataset layout (YOLO detection format):

track/dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

Each label file has one row per object: ``class_id x_center y_center width height``
with normalized coordinates (0..1). For this project class 0 is the tiger.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"
DATA_YAML = DATASET / "data.yaml"


def check_dataset() -> None:
    required = [
        DATASET / "images" / "train",
        DATASET / "images" / "val",
        DATASET / "labels" / "train",
        DATASET / "labels" / "val",
    ]
    missing = [path for path in required if not path.is_dir()]
    if missing:
        paths = "\n".join(f"  - {p.relative_to(HERE)}" for p in missing)
        raise SystemExit(
            "Folder dataset belum lengkap. Buat struktur folder berikut, isi foto "
            "dan label YOLO, lalu jalankan lagi:\n" + paths
        )

    train_images = [p for p in (DATASET / "images" / "train").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    val_images = [p for p in (DATASET / "images" / "val").iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not train_images or not val_images:
        raise SystemExit("Tambahkan foto beranotasi ke images/train dan images/val.")
    for image in train_images + val_images:
        label = DATASET / "labels" / image.parent.name / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"Label tidak ditemukan untuk {image.relative_to(HERE)}: {label.relative_to(HERE)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model deteksi fauna dengan Ultralytics YOLO.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Checkpoint awal Ultralytics (default: yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Contoh: 0 untuk GPU pertama, atau cpu.")
    args = parser.parse_args()

    check_dataset()
    DATA_YAML.write_text(
        "# Dibuat otomatis oleh train_model.py\n"
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: harimau_sumatra\n",
        encoding="utf-8",
    )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics belum terpasang. Jalankan: python -m pip install -r track/requirements.txt") from exc

    model = YOLO(args.weights)
    options = {
        "data": str(DATA_YAML),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(HERE / "runs"),
        "name": "harimau_sumatra",
        "exist_ok": True,
    }
    if args.device:
        options["device"] = args.device
    result = model.train(**options)
    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    exported_model = YOLO(str(best_weights)).export(format="onnx", imgsz=args.imgsz, nms=True)
    browser_model = HERE / "harimau_sumatra.onnx"
    shutil.copy2(exported_model, browser_model)
    print(f"Model training: {best_weights}")
    print(f"Model browser: {browser_model}")


if __name__ == "__main__":
    main()
