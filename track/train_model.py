"""Train a YOLO object detector for the fauna camera tracker.

Expected dataset layout (YOLO detection format):

track/dataset/
  images/train/*.jpg
  images/val/*.jpg
  labels/train/*.txt
  labels/val/*.txt

Each label file has one row per object: ``class_id x_center y_center width height``
with normalized coordinates (0..1). Class 0 is the Sumatran tiger and class 1 is the Sumatran rhino.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE.parent / ".ultralytics"))
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

    for split in ("train", "val"):
        class_ids = set()
        for label in (DATASET / "labels" / split).glob("*.txt"):
            for row in label.read_text(encoding="utf-8").splitlines():
                if row.strip():
                    class_ids.add(row.split()[0])
        missing_classes = {"0", "1"} - class_ids
        if missing_classes:
            names = {"0": "harimau_sumatra", "1": "badak_sumatra"}
            missing_names = ", ".join(names[class_id] for class_id in sorted(missing_classes))
            raise SystemExit(
                f"Dataset {split} belum memiliki anotasi untuk: {missing_names}. "
                "Tambahkan foto dan kotak YOLO untuk kedua kelas sebelum training."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih model deteksi fauna dengan Ultralytics YOLO.")
    parser.add_argument("--weights", default="yolo11n.pt", help="Checkpoint awal Ultralytics (default: yolo11n.pt).")
    parser.add_argument("--epochs", type=int, default=150)
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
        "  0: harimau_sumatra\n"
        "  1: badak_sumatra\n",
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
        "workers": 0,  # Hindari shared memory error 1455 di Windows
        "project": str(HERE / "runs"),
        "name": "badak_harimau",
        "exist_ok": True,
        # Augmentasi agresif untuk dataset kecil
        "augment": True,
        "hsv_h": 0.02,
        "hsv_s": 0.7,
        "hsv_v": 0.5,
        "degrees": 15,
        "translate": 0.2,
        "scale": 0.6,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.3,
        "copy_paste": 0.3,
    }
    import torch
    if args.device is not None:
        options["device"] = args.device
    elif torch.cuda.is_available():
        options["device"] = 0
        print(f"Menggunakan Akselerasi GPU: {torch.cuda.get_device_name(0)}")
    else:
        options["device"] = "cpu"
        print("Menggunakan CPU")
    result = model.train(**options)
    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    exported_model = YOLO(str(best_weights)).export(format="onnx", imgsz=args.imgsz, nms=True)
    browser_model = HERE / "badak_harimau.onnx"
    shutil.copy2(exported_model, browser_model)
    shutil.copy2(exported_model, HERE / "harimau_sumatra.onnx")
    print(f"Model training: {best_weights}")
    print(f"Model browser: {browser_model}")
    print(f"Model fallback: {HERE / 'harimau_sumatra.onnx'}")


if __name__ == "__main__":
    main()
