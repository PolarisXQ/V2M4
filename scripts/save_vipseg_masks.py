from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save the first, middle, and last masked frames for every instance in the "
            "VIPSeg dataset. Different instances in the same video are stored in separate folders."
        )
    )
    parser.add_argument(
        "--vipseg-dataset-path",
        type=str,
        default="/inspurfs/group/mayuexin/yaoych/LP/Dataset/VIPSeg",
        help="Root directory of the VIPSeg dataset.",
    )
    parser.add_argument(
        "--vspw-dataset-path",
        type=str,
        default="/inspurfs/group/mayuexin/yaoych/LP/Dataset/VSPW/VSPW/data",
        help="Root directory of the VSPW dataset (for images).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/inspurfs/group/mayuexin/xiaqi/vipseg/masked_samples",
        help="Directory where the masked frames will be written.",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="all",
        choices=["train", "val", "all"],
        help="Dataset split to process.",
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


def load_mask_as_panoptic(mask_path: str) -> np.ndarray:
    """
    Load mask and convert to panoptic format (instance IDs).
    
    Args:
        mask_path: Path to the mask file.
    
    Returns:
        Panoptic mask array with instance IDs.
    """
    with Image.open(mask_path) as mask_img:
        mask_array = np.array(mask_img)
    
    # 处理不同的掩码格式
    if mask_array.ndim == 3:
        # RGB掩码，转换为实例ID
        gt_pan = np.uint32(mask_array)
        pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
    else:
        # 灰度掩码，直接使用
        pan_gt = mask_array.astype(np.uint32)
    
    return pan_gt


def get_sequence_names(
    vipseg_dataset_path: str,
    vspw_dataset_path: str,
    split_name: str
) -> List[str]:
    """
    Get list of sequence names to process.
    
    Args:
        vipseg_dataset_path: Path to VIPSeg dataset.
        vspw_dataset_path: Path to VSPW dataset.
        split_name: Split name ("train", "val", or "all").
    
    Returns:
        List of sequence names.
    """
    mask_path = os.path.join(vipseg_dataset_path, "panomasks")
    
    if split_name == "all":
        split_folder_names = glob.glob(os.path.join(mask_path, "*"))
        split_folder_names = [os.path.basename(cls) for cls in split_folder_names]
        # Only keep sequences that have map_dict.json
        split_folder_names = [
            cls for cls in split_folder_names 
            if os.path.exists(os.path.join(vspw_dataset_path, cls, "map_dict.json"))
        ]
    else:
        # Load from split file
        split_file = os.path.join(vipseg_dataset_path, f"{split_name}.txt")
        if not os.path.exists(split_file):
            print(f"Warning: Split file {split_file} not found, using all sequences")
            return get_sequence_names(vipseg_dataset_path, vspw_dataset_path, "all")
        
        with open(split_file, "r") as f:
            split_folder_names = f.read().splitlines()
        
        # Filter to only sequences that exist and have map_dict.json
        split_folder_names = [
            cls for cls in split_folder_names
            if os.path.exists(os.path.join(mask_path, cls))
            and os.path.exists(os.path.join(vspw_dataset_path, cls, "map_dict.json"))
        ]
    
    return sorted(split_folder_names)


def find_instance_frames(
    mask_files: List[str],
    instance_id: int
) -> List[Tuple[int, str]]:
    """
    Find all frames where a specific instance appears.
    
    Args:
        mask_files: List of mask file paths.
        instance_id: Instance ID to search for.
    
    Returns:
        List of (frame_index, mask_path) tuples where the instance appears.
    """
    instance_frames = []
    
    for idx, mask_path in enumerate(mask_files):
        pan_gt = load_mask_as_panoptic(mask_path)
        if instance_id in pan_gt:
            instance_frames.append((idx, mask_path))
    
    return instance_frames


def get_image_path_from_mask_path(
    mask_path: str,
    vspw_dataset_path: str,
    map_dict: Dict[str, int]
) -> str:
    """
    Get the corresponding image path from a mask path using map_dict.
    
    Args:
        mask_path: Path to mask file.
        vspw_dataset_path: Path to VSPW dataset root.
        map_dict: Dictionary mapping mask indices to image indices.
    
    Returns:
        Path to corresponding image file.
    """
    seq_name = os.path.basename(os.path.dirname(mask_path))
    mask_index = str(int(os.path.basename(mask_path).split(".")[0]))
    
    if mask_index not in map_dict:
        return None
    
    image_index = map_dict[mask_index]
    image_filename = f"{image_index:08d}.jpg"
    image_path = os.path.join(vspw_dataset_path, seq_name, "origin", image_filename)
    
    return image_path if os.path.exists(image_path) else None


def main() -> None:
    args = parse_args()
    
    output_root = Path(args.output_dir).resolve()
    ensure_dir(output_root)
    
    vipseg_dataset_path = Path(args.vipseg_dataset_path).resolve()
    vspw_dataset_path = Path(args.vspw_dataset_path).resolve()
    mask_path = os.path.join(str(vipseg_dataset_path), "panomasks")
    
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Could not find masks under {mask_path}")
    
    # Get sequence names to process
    sequence_names = get_sequence_names(
        str(vipseg_dataset_path),
        str(vspw_dataset_path),
        args.split_name
    )
    
    print(f"Found {len(sequence_names)} sequences to process")
    
    total_instances = 0
    
    for seq_name in tqdm(sequence_names, desc="Processing sequences"):
        seq_mask_dir = os.path.join(mask_path, seq_name)
        if not os.path.exists(seq_mask_dir):
            print(f"Skipping {seq_name}: mask directory not found")
            continue
        
        # Load map_dict
        map_dict_path = os.path.join(str(vspw_dataset_path), seq_name, "map_dict.json")
        if not os.path.exists(map_dict_path):
            print(f"Skipping {seq_name}: map_dict.json not found")
            continue
        
        with open(map_dict_path, "r") as f:
            map_dict = json.load(f)
        
        # Get all mask files
        mask_files = sorted(glob.glob(os.path.join(seq_mask_dir, "*.png")))
        if not mask_files:
            continue
        
        # Find all unique instances across all masks
        all_instances = set()
        for mask_file in mask_files:
            pan_gt = load_mask_as_panoptic(mask_file)
            unique_ids = np.unique(pan_gt)
            unique_ids = unique_ids[unique_ids > 0]  # Exclude background
            all_instances.update(unique_ids.tolist())
        
        # Process each instance
        for instance_id in all_instances:
            # Find all frames where this instance appears
            instance_frames = find_instance_frames(mask_files, instance_id)
            
            if not instance_frames:
                continue
            
            # Select first, middle, and last frames
            frame_indices = {
                "first": 0,
                "mid": len(instance_frames) // 2,
                "last": len(instance_frames) - 1,
            }
            
            # Create output directory: output_root/seq_name/instance_id/
            instance_output_dir = output_root / seq_name / str(instance_id)
            ensure_dir(instance_output_dir)
            
            # Process selected frames
            for label, frame_idx in frame_indices.items():
                mask_idx, mask_path = instance_frames[frame_idx]
                
                # Get corresponding image path
                image_path = get_image_path_from_mask_path(
                    mask_path,
                    str(vspw_dataset_path),
                    map_dict
                )
                
                if image_path is None or not os.path.exists(image_path):
                    print(f"Skipping {seq_name} instance {instance_id} {label}: missing image")
                    continue
                
                # Load image and mask
                with Image.open(image_path) as img:
                    img_arr = np.array(img)
                    img_h, img_w = img_arr.shape[:2]
                
                pan_gt = load_mask_as_panoptic(mask_path)
                
                # Extract binary mask for this instance
                mask_binary = (pan_gt == instance_id)
                
                # Resize mask to match image dimensions if needed
                mask_h, mask_w = mask_binary.shape[:2]
                if mask_h != img_h or mask_w != img_w:
                    # Resize mask to match image size using nearest neighbor
                    mask_img = Image.fromarray(mask_binary.astype(np.uint8) * 255)
                    mask_img = mask_img.resize((img_w, img_h), Image.Resampling.NEAREST)
                    mask_binary = np.array(mask_img) > 0
                
                # Create masked image
                masked_img = to_masked_image(img_arr, mask_binary)
                
                # Save with mask filename stem
                mask_stem = os.path.basename(mask_path).split(".")[0]
                output_filename = f"{mask_stem}.png"
                masked_img.save(instance_output_dir / output_filename)
            
            total_instances += 1
    
    print(f"Saved masked frames for {total_instances} instances to {output_root}")


if __name__ == "__main__":
    main()

