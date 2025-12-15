import argparse
from pathlib import Path

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python (cv2) is required. Install it with: pip install opencv-python"
    )

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "numpy is required. Install it with: pip install numpy"
    )

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        if desc:
            print(desc)
        return iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create videos from JPEG frames in the DAVIS-2017 dataset with red bounding boxes "
            "around masked objects."
        )
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/inspurfs/group/mayuexin/yaoych/LP/Dataset/vots2022-davis-format",
        help="Root directory of the DAVIS-2017 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/inspurfs/group/mayuexin/xiaqi/vots2022/video_with_boxes",
        help="Directory where the videos will be saved.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output videos.",
    )
    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Thickness of the bounding box lines.",
    )
    return parser.parse_args()


def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def get_bounding_box(mask):
    """
    Get the bounding box of non-zero pixels in a mask.
    
    Args:
        mask: Binary mask array (H, W) where non-zero values indicate the object.
    
    Returns:
        (x1, y1, x2, y2) bounding box coordinates, or None if no object found.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    
    # Convert to binary mask
    mask_binary = mask > 0
    
    # Find non-zero pixels
    coords = np.column_stack(np.where(mask_binary))
    
    if len(coords) == 0:
        return None
    
    # Get bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return (int(x_min), int(y_min), int(x_max), int(y_max))


def draw_bounding_box(image, bbox, color=(0, 0, 255), thickness=2):
    """
    Draw a bounding box on an image.
    
    Args:
        image: BGR image array (H, W, 3).
        bbox: (x1, y1, x2, y2) bounding box coordinates.
        color: BGR color tuple for the box (default: red).
        thickness: Thickness of the box lines.
    
    Returns:
        Image with bounding box drawn.
    """
    if bbox is None:
        return image
    
    x1, y1, x2, y2 = bbox
    # Draw rectangle (cv2 uses BGR format, so (0, 0, 255) is red)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image


def iter_class_dirs(dataset_root):
    """Iterate over class directories in the Annotations folder."""
    annotation_root = dataset_root / "Annotations" / "Full-Resolution"
    if not annotation_root.exists():
        raise FileNotFoundError(f"Could not find annotations under {annotation_root}")
    for path in sorted(p for p in annotation_root.iterdir() if p.is_dir()):
        yield path


def create_video_with_boxes(image_dir, mask_dir, output_path, fps=30, thickness=2):
    """
    Create a video from JPEG frames with red bounding boxes around masked objects.
    
    Args:
        image_dir: Directory containing JPEG frames (e.g., 00000.jpg, 00001.jpg, ...)
        mask_dir: Directory containing mask PNG files (e.g., 00000.png, 00001.png, ...)
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video
        thickness: Thickness of the bounding box lines
    
    Returns:
        True if successful, False otherwise
    """
    # Get all mask files and sort them
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        return False
    
    # Read first frame to get dimensions
    first_mask_path = mask_files[0]
    first_mask_stem = first_mask_path.stem
    first_image_path = image_dir / f"{first_mask_stem}.jpg"
    
    if not first_image_path.exists():
        return False
    
    first_frame = cv2.imread(str(first_image_path))
    if first_frame is None:
        return False
    
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False
    
    # Process all frames
    for mask_file in mask_files:
        mask_stem = mask_file.stem
        image_path = image_dir / f"{mask_stem}.jpg"
        
        if not image_path.exists():
            print(f"Warning: Missing image {image_path} for mask {mask_file}")
            continue
        
        # Read image and mask
        frame = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        
        if frame is None:
            print(f"Warning: Failed to read frame {image_path}")
            continue
        
        if mask is None:
            print(f"Warning: Failed to read mask {mask_file}")
            # Write frame without box if mask is missing
            out.write(frame)
            continue
        
        # Get bounding box from mask
        bbox = get_bounding_box(mask)
        
        # Draw bounding box on frame
        if bbox is not None:
            frame = draw_bounding_box(frame, bbox, color=(0, 0, 255), thickness=thickness)
        
        out.write(frame)
    
    out.release()
    return True


def main():
    args = parse_args()

    output_root = Path(args.output_dir).resolve()
    ensure_dir(output_root)

    dataset_root = Path(args.dataset_path).resolve()
    image_root = dataset_root / "JPEGImages" / "Full-Resolution"
    annotation_root = dataset_root / "Annotations" / "Full-Resolution"
    
    if not image_root.exists():
        raise FileNotFoundError(f"Could not find images under {image_root}")
    if not annotation_root.exists():
        raise FileNotFoundError(f"Could not find annotations under {annotation_root}")
    
    class_dirs = list(iter_class_dirs(dataset_root))
    created_count = 0
    
    for class_dir in tqdm(class_dirs, desc="Creating videos with bounding boxes"):
        class_name = class_dir.name
        
        # Get corresponding image directory
        image_dir = image_root / class_name
        if not image_dir.exists():
            print(f"Skipping {class_name}: missing image directory {image_dir}")
            continue
        
        # Create output video path
        output_path = output_root / f"{class_name}.mp4"
        
        try:
            if create_video_with_boxes(
                image_dir, 
                class_dir, 
                output_path, 
                args.fps,
                args.thickness
            ):
                created_count += 1
            else:
                print(f"Failed to create video for {class_name}")
        except Exception as e:
            print(f"Error creating video for {class_name}: {e}")

    print(f"Created {created_count} videos with bounding boxes in {output_root}")


if __name__ == "__main__":
    main()


