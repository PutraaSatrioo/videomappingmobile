"""Generate augmented copies of training images to expand a tiny dataset.

Requires: Pillow (``pip install Pillow``).

This script creates transformed copies of every image in ``dataset/images/train``
(and matching labels) so YOLO has more variety to learn from.  Run it **once**
before ``train_model.py``.

Each source image produces several variants: horizontal flip, brightness shifts,
small rotations, and crops with the tiger at different positions.  Labels are
adjusted accordingly.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    sys.exit("Pillow belum terpasang. Jalankan: pip install Pillow")

HERE = Path(__file__).resolve().parent
TRAIN_IMAGES = HERE / "dataset" / "images" / "train"
TRAIN_LABELS = HERE / "dataset" / "labels" / "train"

random.seed(42)


def read_label(path: Path) -> list[list[float]]:
    """Return list of [cls, cx, cy, w, h]."""
    rows = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts])
    return rows


def write_label(path: Path, rows: list[list[float]]) -> None:
    lines = []
    for r in rows:
        cls = int(r[0])
        coords = " ".join(f"{v:.4f}" for v in r[1:])
        lines.append(f"{cls} {coords}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def flip_h(img: Image.Image, labels: list[list[float]]):
    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    new_labels = []
    for cls, cx, cy, w, h in labels:
        new_labels.append([cls, 1.0 - cx, cy, w, h])
    return new_img, new_labels


def adjust_brightness(img: Image.Image, labels, factor: float):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor), [row[:] for row in labels]


def adjust_contrast(img: Image.Image, labels, factor: float):
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor), [row[:] for row in labels]


def add_blur(img: Image.Image, labels, radius: float = 1.5):
    return img.filter(ImageFilter.GaussianBlur(radius=radius)), [row[:] for row in labels]


def add_color_shift(img: Image.Image, labels, factor: float):
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor), [row[:] for row in labels]


def resize_with_pad(img: Image.Image, labels, scale: float):
    """Shrink image into a padded canvas (simulating tiger at distance)."""
    orig_w, orig_h = img.size
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    small = img.resize((new_w, new_h), Image.LANCZOS)

    # Random offset
    max_x = orig_w - new_w
    max_y = orig_h - new_h
    ox = random.randint(0, max(0, max_x))
    oy = random.randint(0, max(0, max_y))

    canvas = Image.new("RGB", (orig_w, orig_h), (random.randint(0, 40), random.randint(0, 40), random.randint(0, 40)))
    canvas.paste(small, (ox, oy))

    new_labels = []
    for cls, cx, cy, w, h in labels:
        ncx = clamp((cx * new_w + ox) / orig_w)
        ncy = clamp((cy * new_h + oy) / orig_h)
        nw = clamp(w * scale)
        nh = clamp(h * scale)
        if nw > 0.01 and nh > 0.01:
            new_labels.append([cls, ncx, ncy, nw, nh])
    return canvas, new_labels


def main():
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    originals = sorted(
        p for p in TRAIN_IMAGES.iterdir() if p.suffix.lower() in extensions
    )

    if not originals:
        sys.exit("Tidak ada gambar training ditemukan.")

    count = 0
    for img_path in originals:
        label_path = TRAIN_LABELS / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"Skip {img_path.name} — label tidak ada")
            continue

        img = Image.open(img_path).convert("RGB")
        labels = read_label(label_path)
        if not labels:
            continue

        base = img_path.stem
        augmentations = [
            ("flip", flip_h(img, labels)),
            ("bright08", adjust_brightness(img, labels, 0.75)),
            ("bright12", adjust_brightness(img, labels, 1.3)),
            ("contrast", adjust_contrast(img, labels, 1.4)),
            ("blur", add_blur(img, labels, 2.0)),
            ("color", add_color_shift(img, labels, 0.6)),
            ("scale07", resize_with_pad(img, labels, 0.7)),
            ("scale05", resize_with_pad(img, labels, 0.5)),
        ]

        # Also combine flip + brightness
        flipped_img, flipped_labels = flip_h(img, labels)
        augmentations.append(
            ("flip_bright", adjust_brightness(flipped_img, flipped_labels, 0.8))
        )
        augmentations.append(
            ("flip_scale", resize_with_pad(flipped_img, flipped_labels, 0.6))
        )

        for suffix, (aug_img, aug_labels) in augmentations:
            if not aug_labels:
                continue
            new_name = f"{base}_aug_{suffix}"
            aug_img.save(TRAIN_IMAGES / f"{new_name}.jpg", quality=92)
            write_label(TRAIN_LABELS / f"{new_name}.txt", aug_labels)
            count += 1

    print(f"Selesai: {count} gambar augmentasi ditambahkan ke {TRAIN_IMAGES}")
    print(f"Total gambar training sekarang: {len(list(TRAIN_IMAGES.iterdir()))}")


if __name__ == "__main__":
    main()
