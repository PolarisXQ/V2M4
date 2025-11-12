from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save the first, middle, and last masked frames for every class in the "
            "DAVIS-2017 dataset without using the project dataloader."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/inspurfs/group/mayuexin/yaoych/LP/Dataset/DAVIS-2017",
        help="Root directory of the DAVIS-2017 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/inspurfs/group/mayuexin/xiaqi/DAVIS-2017/masked_samples",
        help="Directory where the masked frames will be written.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def to_masked_image(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    """
    Combines a color image and a binary mask into a masked RGB image.

    Args:
        image: RGB image as uint8 array of shape (H, W, 3).
        mask: Boolean mask (H, W) where True keeps the pixel.
    """
    if mask.dtype != np.bool_:
        mask = mask.astype(bool)

    masked = mask[..., None] * image
    return Image.fromarray(masked.astype(np.uint8))


def iter_class_dirs(dataset_root: Path) -> Iterable[Path]:
    annotation_root = dataset_root / "Annotations" / "Full-Resolution"
    if not annotation_root.exists():
        raise FileNotFoundError(f"Could not find annotations under {annotation_root}")
    for path in sorted(p for p in annotation_root.iterdir() if p.is_dir()):
        yield path


def main() -> None:
    args = parse_args()

    output_root = Path(args.output_dir).resolve()
    ensure_dir(output_root)

    dataset_root = Path(args.dataset_path).resolve()
    image_root = dataset_root / "JPEGImages" / "Full-Resolution"

    class_dirs = list(iter_class_dirs(dataset_root))
    for class_dir in tqdm(class_dirs, desc="Classes"):
        class_name = class_dir.name
        mask_files = sorted(class_dir.glob("*.png"))
        if not mask_files:
            continue

        image_dir = image_root / class_name
        if not image_dir.exists():
            print(f"Skipping {class_name}: missing image directory {image_dir}")
            continue

        frame_indices = {
            "first": 0,
            "mid": len(mask_files) // 2,
            "last": len(mask_files) - 1,
        }

        class_output_dir = output_root / class_name
        ensure_dir(class_output_dir)

        for label, idx in frame_indices.items():
            mask_path = mask_files[idx]
            image_path = image_dir / (mask_path.stem + ".jpg")

            if not image_path.exists():
                print(f"Skipping {class_name} {label}: missing image {image_path}")
                continue

            with Image.open(image_path) as img, Image.open(mask_path) as mask:
                img_arr = np.array(img)
                mask_arr = np.array(mask)

                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]

                mask_binary = mask_arr > 0
                masked_img = to_masked_image(img_arr, mask_binary)
                masked_img.save(class_output_dir / f"{mask_path.stem}.png")

    print(f"Saved masked frames to {output_root}")


if __name__ == "__main__":
    main()

