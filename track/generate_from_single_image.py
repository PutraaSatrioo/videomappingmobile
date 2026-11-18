"""Generate a synthetic YOLO dataset from a single transparent PNG image.

This script takes a single transparent cutout (e.g., Sumatra_Harimau.png),
and generates varied training and validation images with exact bounding-box labels:
- Random scale (distant to close-up)
- Random rotation (-20° to +20°)
- Horizontal flip (left/right facing)
- Lighting, contrast, brightness, and camera blur variations
- Varied realistic backgrounds (Unity-style sky-ground gradient, dark rooms, monitor displays)
- Automatic exact YOLO bounding box calculation (0 x y w h)
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE / "harimau_sumatra" / "Sumatra_Harimau.png"
DATASET = HERE / "dataset"


def create_gradient_background(width: int, height: int, mode: str) -> Image.Image:
    """Create simulated realistic backgrounds (Unity scene, monitor, dark room)."""
    base = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(base)

    if mode == "unity_scene":
        # Sky (blueish/gray) to ground (light brown/gray) like Unity Editor viewport
        sky_r, sky_g, sky_b = random.randint(140, 190), random.randint(160, 210), random.randint(190, 240)
        ground_r, ground_g, ground_b = random.randint(110, 160), random.randint(95, 140), random.randint(80, 120)
        horizon = int(height * random.uniform(0.48, 0.62))

        for y in range(height):
            if y < horizon:
                ratio = y / max(1, horizon)
                r = int(sky_r * (1 - ratio * 0.25))
                g = int(sky_g * (1 - ratio * 0.25))
                b = int(sky_b * (1 - ratio * 0.25))
            else:
                ratio = (y - horizon) / max(1, height - horizon)
                r = int(ground_r + ratio * 20)
                g = int(ground_g + ratio * 20)
                b = int(ground_b + ratio * 20)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

    elif mode == "dark_scene":
        # Dark room or night background
        top_val = random.randint(8, 28)
        bot_val = random.randint(20, 50)
        for y in range(height):
            ratio = y / height
            v = int(top_val + (bot_val - top_val) * ratio)
            draw.line([(0, y), (width, y)], fill=(v, v, int(v * 1.15)))

    elif mode == "monitor_desktop":
        # Desktop monitor look (dark UI panels, editor gray)
        bg_col = random.randint(35, 70)
        base.paste((bg_col, bg_col, bg_col + 5), [0, 0, width, height])
        # Add random horizontal/vertical panel lines like Unity layout
        for _ in range(random.randint(1, 3)):
            line_y = random.randint(20, height - 20)
            draw.line([(0, line_y), (width, line_y)], fill=(bg_col - 18, bg_col - 18, bg_col - 18), width=random.randint(1, 3))
    else:
        # Neutral gray/indoor tone
        r = random.randint(50, 140)
        g = int(r * random.uniform(0.9, 1.05))
        b = int(r * random.uniform(0.95, 1.1))
        base.paste((r, g, b), [0, 0, width, height])

    return base


def augment_tiger(tiger: Image.Image) -> tuple[Image.Image, bool]:
    """Apply random photorealistic variations to the tiger cutout."""
    img = tiger.copy()
    is_flipped = False

    # 1. Flip horizontal (50% chance)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        is_flipped = True

    # 2. Random rotation (-18° to +18°)
    angle = random.uniform(-18, 18)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=True)

    # 3. Brightness variation (0.65 to 1.35)
    b_factor = random.uniform(0.65, 1.35)
    img_rgb = img.convert("RGB")
    alpha = img.split()[-1]
    img_rgb = ImageEnhance.Brightness(img_rgb).enhance(b_factor)

    # 4. Contrast variation (0.8 to 1.3)
    c_factor = random.uniform(0.8, 1.3)
    img_rgb = ImageEnhance.Contrast(img_rgb).enhance(c_factor)

    # 5. Color saturation (keep galaxy tones, factor 0.8 to 1.3)
    s_factor = random.uniform(0.8, 1.3)
    img_rgb = ImageEnhance.Color(img_rgb).enhance(s_factor)

    # 6. Slight camera blur (simulating phone camera focus/motion)
    if random.random() < 0.4:
        img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.6)))

    # Recombine with alpha
    img = Image.merge("RGBA", (*img_rgb.split(), alpha))
    return img, is_flipped


def generate_single_sample(
    tiger_cropped: Image.Image,
    canvas_w: int = 640,
    canvas_h: int = 640,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Synthesize one image and return (canvas_image, (x_center, y_center, width, height))."""
    # Pick a background type
    bg_type = random.choices(
        ["unity_scene", "monitor_desktop", "dark_scene", "neutral"],
        weights=[0.45, 0.25, 0.15, 0.15],
    )[0]
    canvas = create_gradient_background(canvas_w, canvas_h, bg_type)

    # Augment tiger
    aug_tiger, _ = augment_tiger(tiger_cropped)

    # Target scale: tiger width should be between 25% and 80% of canvas
    target_width_ratio = random.uniform(0.28, 0.78)
    target_w = int(canvas_w * target_width_ratio)
    aspect = aug_tiger.height / max(1, aug_tiger.width)
    target_h = int(target_w * aspect)

    # Keep within canvas bounds
    if target_h > int(canvas_h * 0.85):
        target_h = int(canvas_h * 0.85)
        target_w = int(target_h / aspect)

    aug_tiger = aug_tiger.resize((target_w, target_h), Image.LANCZOS)

    # Placement coordinates (keep mostly visible, biased slightly toward center)
    max_x = canvas_w - target_w
    max_y = canvas_h - target_h

    if max_x > 0:
        # Bias placement toward center to match bow reticle
        mean_x = max_x / 2.0
        paste_x = int(random.gauss(mean_x, max_x * 0.22))
        paste_x = max(0, min(max_x, paste_x))
    else:
        paste_x = 0

    if max_y > 0:
        mean_y = max_y / 2.0
        paste_y = int(random.gauss(mean_y, max_y * 0.22))
        paste_y = max(0, min(max_y, paste_y))
    else:
        paste_y = 0

    # Paste onto canvas
    canvas.paste(aug_tiger, (paste_x, paste_y), aug_tiger)

    # Calculate actual tight bounding box from the pasted alpha
    alpha = aug_tiger.split()[-1]
    bbox = alpha.getbbox()
    if bbox:
        bx1, by1, bx2, by2 = bbox
        x1 = paste_x + bx1
        y1 = paste_y + by1
        x2 = paste_x + bx2
        y2 = paste_y + by2
    else:
        x1, y1 = paste_x, paste_y
        x2, y2 = paste_x + target_w, paste_y + target_h

    # Convert to normalized YOLO format (0..1)
    cx = ((x1 + x2) / 2.0) / canvas_w
    cy = ((y1 + y2) / 2.0) / canvas_h
    w = (x2 - x1) / canvas_w
    h = (y2 - y1) / canvas_h

    # Clamping
    cx = max(0.01, min(0.99, cx))
    cy = max(0.01, min(0.99, cy))
    w = max(0.02, min(0.98, w))
    h = max(0.02, min(0.98, h))

    return canvas, (cx, cy, w, h)


def build_synthetic_dataset(
    source_path: Path,
    num_train: int = 120,
    num_val: int = 30,
    clean: bool = True,
) -> None:
    if not source_path.is_file():
        raise SystemExit(f"Source file not found: {source_path}")

    print(f"Loading source cutout: {source_path.name}")
    raw_img = Image.open(source_path).convert("RGBA")
    bbox = raw_img.getbbox()
    if not bbox:
        raise SystemExit("Source image has no non-transparent pixels!")
    tiger_cropped = raw_img.crop(bbox)
    print(f"Cropped tiger size: {tiger_cropped.size}")

    train_img_dir = DATASET / "images" / "train"
    val_img_dir = DATASET / "images" / "val"
    train_lbl_dir = DATASET / "labels" / "train"
    val_lbl_dir = DATASET / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        if clean:
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()

    print(f"Generating {num_train} training images...")
    random.seed(42)
    for i in range(1, num_train + 1):
        canvas, (cx, cy, w, h) = generate_single_sample(tiger_cropped, 640, 640)
        name = f"tiger_syn_train_{i:03d}"
        canvas.save(train_img_dir / f"{name}.jpg", quality=90)
        (train_lbl_dir / f"{name}.txt").write_text(
            f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n",
            encoding="utf-8",
        )

    print(f"Generating {num_val} validation images...")
    for i in range(1, num_val + 1):
        canvas, (cx, cy, w, h) = generate_single_sample(tiger_cropped, 640, 640)
        name = f"tiger_syn_val_{i:03d}"
        canvas.save(val_img_dir / f"{name}.jpg", quality=90)
        (val_lbl_dir / f"{name}.txt").write_text(
            f"0 {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n",
            encoding="utf-8",
        )

    # Ensure data.yaml is created
    data_yaml = DATASET / "data.yaml"
    data_yaml.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: harimau_sumatra\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print(f"SUKSES! Dataset sintetis berhasil dibuat:")
    print(f"  - Training: {num_train} foto + label di {train_img_dir}")
    print(f"  - Validasi: {num_val} foto + label di {val_img_dir}")
    print(f"  - Konfigurasi: {data_yaml}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset from 1 transparent PNG.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path ke file PNG transparan.")
    parser.add_argument("--train-count", type=int, default=120, help="Jumlah gambar train (default: 120).")
    parser.add_argument("--val-count", type=int, default=30, help="Jumlah gambar val (default: 30).")
    args = parser.parse_args()

    build_synthetic_dataset(args.source, args.train_count, args.val_count)


if __name__ == "__main__":
    main()
