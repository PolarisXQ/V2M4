import argparse
from pathlib import Path

try:
    import cv2
except ImportError:
    raise ImportError(
        "opencv-python (cv2) is required. Install it with: pip install opencv-python"
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
            "Create videos from JPEG frames in the DAVIS-2017 dataset."
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
        default="/inspurfs/group/mayuexin/xiaqi/vots2022/video",
        help="Directory where the videos will be saved.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for the output videos.",
    )
    return parser.parse_args()


def ensure_dir(path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def iter_class_dirs(dataset_root):
    """Iterate over class directories in the JPEGImages folder."""
    image_root = dataset_root / "JPEGImages" / "Full-Resolution"
    if not image_root.exists():
        raise FileNotFoundError(f"Could not find images under {image_root}")
    for path in sorted(p for p in image_root.iterdir() if p.is_dir()):
        yield path


def create_video_from_frames(frame_dir, output_path, fps=30):
    """
    Create a video from JPEG frames in a directory.
    
    Args:
        frame_dir: Directory containing JPEG frames (e.g., 00000.jpg, 00001.jpg, ...)
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video
    """
    # Get all JPEG frames and sort them
    frame_files = sorted(frame_dir.glob("*.jpg"))
    if not frame_files:
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        return False
    
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Failed to open video writer for {output_path}")
        return False
    
    # Write all frames
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            print(f"Warning: Failed to read frame {frame_file}")
            continue
        out.write(frame)
    
    out.release()
    return True


def main():
    args = parse_args()

    output_root = Path(args.output_dir).resolve()
    ensure_dir(output_root)

    dataset_root = Path(args.dataset_path).resolve()
    
    class_dirs = list(iter_class_dirs(dataset_root))
    created_count = 0
    
    for class_dir in tqdm(class_dirs, desc="Creating videos"):
        class_name = class_dir.name
        
        # Create output video path
        output_path = output_root / f"{class_name}.mp4"
        
        try:
            if create_video_from_frames(class_dir, output_path, args.fps):
                created_count += 1
            else:
                print(f"Failed to create video for {class_name}")
        except Exception as e:
            print(f"Error creating video for {class_name}: {e}")

    print(f"Created {created_count} videos in {output_root}")


if __name__ == "__main__":
    main()
