"""Generate synthetic YOLO dataset for both Harimau Sumatra and Badak Sumatra.

Supports multi-class dataset synthesis with automatic bounding boxes:
- Class 0: harimau_sumatra
- Class 1: badak_sumatra
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

HERE = Path(__file__).resolve().parent
DATASET = HERE / "dataset"

DEFAULT_CLASSES = {
    0: {
        "name": "harimau_sumatra",
        "path": HERE / "harimau_sumatra" / "Sumatra_Harimau.png",
    },
    1: {
        "name": "badak_sumatra",
        "path": HERE / "badak_sumatra" / "badak_clean.png",
    },
}


def create_gradient_background(width: int, height: int, mode: str) -> Image.Image:
    """Create simulated realistic backgrounds (Unity scene, monitor, dark room)."""
    base = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(base)

    if mode == "unity_scene":
        # Sky to ground like Unity Editor viewport
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
        top_val = random.randint(8, 28)
        bot_val = random.randint(20, 50)
        for y in range(height):
            ratio = y / height
            v = int(top_val + (bot_val - top_val) * ratio)
            draw.line([(0, y), (width, y)], fill=(v, v, int(v * 1.15)))

    elif mode == "monitor_desktop":
        bg_col = random.randint(35, 70)
        base.paste((bg_col, bg_col, bg_col + 5), [0, 0, width, height])
        for _ in range(random.randint(1, 3)):
            line_y = random.randint(20, height - 20)
            draw.line([(0, line_y), (width, line_y)], fill=(bg_col - 18, bg_col - 18, bg_col - 18), width=random.randint(1, 3))
    else:
        r = random.randint(50, 140)
        g = int(r * random.uniform(0.9, 1.05))
        b = int(r * random.uniform(0.95, 1.1))
        base.paste((r, g, b), [0, 0, width, height])

    return base


def augment_animal(cutout: Image.Image) -> tuple[Image.Image, bool]:
    """Apply random photorealistic variations to the animal cutout."""
    img = cutout.copy()
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

    # 5. Color saturation (0.8 to 1.3)
    s_factor = random.uniform(0.8, 1.3)
    img_rgb = ImageEnhance.Color(img_rgb).enhance(s_factor)

    # 6. Slight camera blur
    if random.random() < 0.4:
        img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.6)))

    img = Image.merge("RGBA", (*img_rgb.split(), alpha))
    return img, is_flipped


def generate_single_sample(
    animal_cropped: Image.Image,
    class_id: int,
    canvas_w: int = 640,
    canvas_h: int = 640,
) -> tuple[Image.Image, tuple[int, float, float, float, float]]:
    """Synthesize one image and return (canvas_image, (class_id, cx, cy, w, h))."""
    bg_type = random.choices(
        ["unity_scene", "monitor_desktop", "dark_scene", "neutral"],
        weights=[0.45, 0.25, 0.15, 0.15],
    )[0]
    canvas = create_gradient_background(canvas_w, canvas_h, bg_type)

    aug_animal, _ = augment_animal(animal_cropped)

    target_width_ratio = random.uniform(0.28, 0.78)
    target_w = int(canvas_w * target_width_ratio)
    aspect = aug_animal.height / max(1, aug_animal.width)
    target_h = int(target_w * aspect)

    if target_h > int(canvas_h * 0.85):
        target_h = int(canvas_h * 0.85)
        target_w = int(target_h / aspect)

    aug_animal = aug_animal.resize((target_w, target_h), Image.LANCZOS)

    max_x = canvas_w - target_w
    max_y = canvas_h - target_h

    if max_x > 0:
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

    canvas.paste(aug_animal, (paste_x, paste_y), aug_animal)

    alpha = aug_animal.split()[-1]
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

    cx = ((x1 + x2) / 2.0) / canvas_w
    cy = ((y1 + y2) / 2.0) / canvas_h
    w = (x2 - x1) / canvas_w
    h = (y2 - y1) / canvas_h

    cx = max(0.01, min(0.99, cx))
    cy = max(0.01, min(0.99, cy))
    w = max(0.02, min(0.98, w))
    h = max(0.02, min(0.98, h))

    return canvas, (class_id, cx, cy, w, h)


def build_multi_class_dataset(
    classes: dict[int, dict[str, any]],
    per_class_train: int = 100,
    per_class_val: int = 25,
    clean: bool = True,
) -> None:
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

    loaded_cutouts = {}
    for cid, info in classes.items():
        src = Path(info["path"])
        if not src.is_file():
            raise SystemExit(f"Cutout file not found: {src}")
        img = Image.open(src).convert("RGBA")
        bbox = img.getbbox()
        if not bbox:
            raise SystemExit(f"No non-transparent pixels in {src}")
        loaded_cutouts[cid] = (info["name"], img.crop(bbox))
        print(f"Loaded class {cid} ({info['name']}): size {loaded_cutouts[cid][1].size}")

    random.seed(42)

    total_train = 0
    total_val = 0

    for cid, (name, cropped_img) in loaded_cutouts.items():
        print(f"Generating {per_class_train} training samples for {name} (class {cid})...")
        for i in range(1, per_class_train + 1):
            canvas, (c_id, cx, cy, w, h) = generate_single_sample(cropped_img, cid, 640, 640)
            sample_name = f"{name}_syn_train_{i:03d}"
            canvas.save(train_img_dir / f"{sample_name}.jpg", quality=90)
            (train_lbl_dir / f"{sample_name}.txt").write_text(
                f"{c_id} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n",
                encoding="utf-8",
            )
            total_train += 1

        print(f"Generating {per_class_val} validation samples for {name} (class {cid})...")
        for i in range(1, per_class_val + 1):
            canvas, (c_id, cx, cy, w, h) = generate_single_sample(cropped_img, cid, 640, 640)
            sample_name = f"{name}_syn_val_{i:03d}"
            canvas.save(val_img_dir / f"{sample_name}.jpg", quality=90)
            (val_lbl_dir / f"{sample_name}.txt").write_text(
                f"{c_id} {cx:.5f} {cy:.5f} {w:.5f} {h:.5f}\n",
                encoding="utf-8",
            )
            total_val += 1

    # Write data.yaml with all classes
    names_yaml = "\n".join(f"  {cid}: {info['name']}" for cid, info in classes.items())
    data_yaml = DATASET / "data.yaml"
    data_yaml.write_text(
        f"path: {DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_yaml}\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("SUKSES! Multi-class dataset berhasil dibuat:")
    print(f"  - Total Training: {total_train} foto + label")
    print(f"  - Total Validasi: {total_val} foto + label")
    print(f"  - Kelas terdaftar:")
    for cid, info in classes.items():
        print(f"      {cid}: {info['name']}")
    print(f"  - Config YAML: {data_yaml}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for multiple fauna classes.")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=25)
    args = parser.parse_args()

    build_multi_class_dataset(DEFAULT_CLASSES, args.train_per_class, args.val_per_class)


if __name__ == "__main__":
    main()
