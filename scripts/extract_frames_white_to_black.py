#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import cv2
import numpy as np

# useage:
# python extract_frames_white_to_black.py --input /public/home/xiaqi2025/video/V2M4/example_video/parrot.mp4 --output example2/parrot --prefix '' --start-index 1 --tolerance 0 --interval 1
# python extract_frames_white_to_black.py --input /public/home/xiaqi2025/video/V2M4/example_video/blackswan.mp4 --output example3/blackswan --prefix '' --start-index 1 --tolerance 30 --interval 5


def replace_white_with_black(frame: np.ndarray, tolerance: int = 0) -> np.ndarray:
    if tolerance <= 0:
        mask = (
            (frame[:, :, 0] == 255)
            & (frame[:, :, 1] == 255)
            & (frame[:, :, 2] == 255)
        )
    else:
        # Treat near-white as white when tolerance > 0
        thresh = 255 - tolerance
        mask = (
            (frame[:, :, 0] >= thresh)
            & (frame[:, :, 1] >= thresh)
            & (frame[:, :, 2] >= thresh)
        )
    frame[mask] = 0
    return frame


def extract_frames(
    input_video_path: Path,
    output_dir: Path,
    filename_prefix: str = "frame_",
    start_index: int = 1,
    tolerance: int = 0,
    interval: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    frame_index = start_index
    saved = 0
    current_frame = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Only save frames at the specified interval
            if current_frame % interval == 0:
                # frame = replace_white_with_black(frame, tolerance=tolerance)

                out_name = f"{filename_prefix}{frame_index:06d}.png"
                out_path = output_dir / out_name
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"Failed to write: {out_path}")

                frame_index += 1
                saved += 1
                if saved % 50 == 0:
                    print(f"Saved {saved} frames...")
            
            current_frame += 1
    finally:
        cap.release()

    print(f"Done. Saved {saved} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all frames from a video, convert white pixels to black, and save as PNGs."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Directory to save frames. Defaults to the video directory (same folder as the video)."
        ),
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame_",
        help="Filename prefix for saved frames (default: frame_)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index for frame numbering (default: 1)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=0,
        help=(
            "Treat near-white as white: set 0 for exact white only, >0 for tolerance (e.g., 5)"
        ),
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Extract one frame every n frames (default: 1, extract all frames)",
    )

    args = parser.parse_args()

    input_video_path = Path(args.input).expanduser().resolve()
    if not input_video_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_video_path}")

    if args.output is None:
        output_dir = input_video_path.parent
    else:
        output_dir = Path(args.output).expanduser().resolve()

    extract_frames(
        input_video_path=input_video_path,
        output_dir=output_dir,
        filename_prefix=args.prefix,
        start_index=args.start_index,
        tolerance=args.tolerance,
        interval=args.interval,
    )


if __name__ == "__main__":
    main()


