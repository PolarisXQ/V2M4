from PIL import Image
import os
import numpy as np
from typing import Dict, List, Tuple, Any
import glob
import json
from tqdm import tqdm  # type: ignore


def _compute_bbox_ratio(mask: np.ndarray) -> float:
    bbox = np.where(mask > 0)
    if bbox[0].size > 0:    
        x1, y1, x2, y2 = bbox[0].min(), bbox[1].min(), bbox[0].max(), bbox[1].max()
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        return (w * h / (mask.shape[1] * mask.shape[0])).tolist()
    else:
        return 0.0

def _load_vipseg_categories(categories_file: str) -> Dict[int, str]:
    """加载VIPSeg类别信息"""
    try:
        with open(categories_file, 'r', encoding='utf-8') as f:
            categories = json.load(f)
        return {cat['id']: cat['name'] for cat in categories}
    except Exception as e:
        print(f"加载类别文件失败: {e}")
        return {}

split_names = ["train", "val", "all"]
class VIPSeg_origin_mix_dataloader:
    def __init__(self, vipseg_dataset_path: str=r"/inspurfs/group/mayuexin/yaoych/LP/Dataset/VIPSeg", vspw_dataset_path: str=r"/inspurfs/group/mayuexin/yaoych/LP/Dataset/VSPW/VSPW/data", split_name: str="all", 
    length: int=84, target_resolution: tuple[int, int]=(480, 270), 
    min_instance_ratio: float=0.00, min_bbox_ratio: float=0.00):
        self.vipseg_dataset_path = vipseg_dataset_path
        self.vspw_dataset_path = vspw_dataset_path
        self.mask_path = os.path.join(vipseg_dataset_path, "panomasks")
        self.split_name = split_name
        assert split_name in split_names, "split_name must be in " + str(split_names)
        self.length = length
        self.target_resolution = target_resolution
        self.min_instance_ratio = min_instance_ratio
        self.min_bbox_ratio = min_bbox_ratio
        self.CLASSES = _load_vipseg_categories(os.path.join(vipseg_dataset_path, "VIPSeg_720P", "panoVIPSeg_categories.json"))
        self.CLASSES_INDEX = {v: k for k, v in self.CLASSES.items()}

        #load from json file if exists
        if os.path.exists(os.path.join(self.vipseg_dataset_path, "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json")):
            self.dataset_list = json.load(open(os.path.join(self.vipseg_dataset_path, "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json"), "r"))
            print("data loaded from json file")
            print("total dataset length: ", len(self.dataset_list))
            return
        else:
            print("no json file found")


            split_folder_names = glob.glob(os.path.join(self.mask_path, "*"))
            split_folder_names = [os.path.basename(cls) for cls in split_folder_names]
            split_folder_names = [cls for cls in split_folder_names if os.path.exists(os.path.join(self.vspw_dataset_path, cls, "map_dict.json"))]

            if split_name == "val" or split_name == "train":
                with open(os.path.join(self.vipseg_dataset_path, split_name+".txt"), "r") as f:
                    split_folder_names = f.read().splitlines()
                split_folder_names = [cls for cls in split_folder_names if cls in split_folder_names]




            self.split_folder_names = split_folder_names

            print("preparing data...")
            seq_iter = []
            for split_folder_name in self.split_folder_names:
                seq_name = split_folder_name.split("/")[-1]
                mask_files = glob.glob(os.path.join(self.mask_path, seq_name, "*.png"))
                img_files = glob.glob(os.path.join(self.vspw_dataset_path, seq_name,"origin", "*.jpg"))
                mask_files.sort()
                img_files.sort()
                first_mask_index = str(int(os.path.basename(mask_files[0]).split(".")[0]))
                last_mask_index = str(int(os.path.basename(mask_files[-1]).split(".")[0]))
                map_dict = json.load(open(os.path.join(self.vspw_dataset_path, seq_name, "map_dict.json"), "r"))
                img_files_basename = [os.path.basename(img_file) for img_file in img_files]
                #keep img_files between first_mask_index and last_mask_index(using map_dict)
                first_img_index = img_files_basename.index(str(map_dict[first_mask_index]).zfill(8)+".jpg")
                last_img_index = img_files_basename.index(str(map_dict[last_mask_index]).zfill(8)+".jpg")
                img_files = img_files[first_img_index:last_img_index+1]
                
                
                if len(img_files) < length:
                    continue
                seq_iter.append((seq_name, mask_files, img_files))
            seq_iter =tqdm(seq_iter, desc="preparing data", unit="seq") 
            self.dataset_list = []
            for seq_name, mask_files, img_files in seq_iter:
            # 图像尺寸（优先 JPEGImages）
                map_dict = json.load(open(os.path.join(self.vspw_dataset_path, seq_name, "map_dict.json"), "r"))
                img_files_basename = [os.path.basename(img_file) for img_file in img_files]
                mask_files_basename = [os.path.basename(mask_file) for mask_file in mask_files]
                with Image.open(mask_files[0]) as m0:
                    img_w, img_h = m0.size

                instance_ratios_dict: Dict[int, List[float]] = {}
                instance_start_end: Dict[int, Tuple[int, int]] = {}  # inst_id -> (start_frame of the image, end_frame of the image)
                instance_mask_start_end: Dict[int, Tuple[int, int]] = {}  # inst_id -> (start_frame of the mask, end_frame of the mask)
                bbox_ratios_dict: Dict[int, List[float]] = {} 
                frame_iter = tqdm(mask_files, desc=f'{seq_name} masks', unit='frame', leave=False)

                for mf in frame_iter:
                    with Image.open(mf) as m:
                        
                        mask_array = np.array(m)
                
                        # 处理不同的掩码格式，与create_panoptic_video_labels.py保持一致
                        if mask_array.ndim == 3:
                            # RGB掩码，转换为实例ID（与create_panoptic_video_labels.py第91-92行一致）
                            gt_pan = np.uint32(mask_array)
                            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
                        else:
                            # 灰度掩码，直接使用
                            pan_gt = mask_array.astype(np.uint32)
                        
                        # 找到所有实例ID
                        unique_ids = np.unique(pan_gt)
                        unique_ids = unique_ids[unique_ids > 0]  # 排除背景
                        
                        if img_w <= 0 or img_h <= 0:
                            h, w = pan_gt.shape[:2]
                            denom = float(w * h)
                        else:
                            denom = float(img_w * img_h)
                        if denom <= 0:
                            continue
                        
                        for inst_id in unique_ids.tolist():
                            mask = (pan_gt == inst_id)
                            area = int(mask.sum())
                            if area <= 0:
                                continue
                            ratio = area / denom
                            

                            if inst_id not in instance_ratios_dict:
                                instance_ratios_dict[inst_id] = []
                            instance_ratios_dict[inst_id].append(ratio)
                            
                            #cal the bbox ratio of the mask
                            bbox_ratio = _compute_bbox_ratio(mask)
                            if inst_id not in bbox_ratios_dict:
                                bbox_ratios_dict[inst_id] = []
                            if inst_id not in instance_start_end:
                                current_image = map_dict[str(int(os.path.basename(mf).split(".")[0]))]
                                instance_start_end[inst_id] = (current_image, current_image)
                            else:
                                current_image = map_dict[str(int(os.path.basename(mf).split(".")[0]))]
                                instance_start_end[inst_id] = (min(instance_start_end[inst_id][0], current_image), max(instance_start_end[inst_id][1], current_image))
                            if inst_id not in instance_mask_start_end:
                                instance_mask_start_end[inst_id] = (os.path.basename(mf), os.path.basename(mf))
                            else:
                                instance_mask_start_end[inst_id] = (min(instance_mask_start_end[inst_id][0], os.path.basename(mf)), max(instance_mask_start_end[inst_id][1], os.path.basename(mf)))
                            bbox_ratios_dict[inst_id].append(bbox_ratio)
                            assert bbox_ratio >= ratio

            

            # 计算每个实例的统计信息
                for inst_id in instance_ratios_dict.keys():
                    inst_ratios = instance_ratios_dict[inst_id]
                    bbox_ratios = bbox_ratios_dict[inst_id]
                    if inst_ratios:
                        inst_mean = float(np.mean(inst_ratios))
                        bbox_mean = float(np.mean(bbox_ratios))
                        if inst_mean < self.min_instance_ratio or bbox_mean < self.min_bbox_ratio:
                            continue

                        start_frame, end_frame = instance_start_end[inst_id]
                        if end_frame - start_frame + 1 < self.length:
                            continue
                        # for i in range(start_frame, end_frame - self.length + 1):
                            
                        mask_files_slice = mask_files[mask_files_basename.index(instance_mask_start_end[inst_id][0]):mask_files_basename.index(instance_mask_start_end[inst_id][1])+1]
                        img_files_slice = img_files[img_files_basename.index(str(start_frame).zfill(8)+".jpg"):img_files_basename.index(str(end_frame).zfill(8)+".jpg")+1]

                        #填充mask_files_slice to the length of self.length with None in the middle
                        new_mask_files_slice = []
                        for i in range(len(mask_files_slice) - 1):
                            new_mask_files_slice.append(mask_files_slice[i])
                            new_mask_files_slice.extend([None] * (map_dict[str(int(os.path.basename(mask_files_slice[i+1]).split(".")[0]))]-map_dict[str(int(os.path.basename(mask_files_slice[i]).split(".")[0]))]-1))
                        new_mask_files_slice.append(mask_files_slice[-1])
                        mask_files_slice = new_mask_files_slice

                        if inst_id == 0:
                            continue  # 跳过背景
                        if inst_id < 125:
                            semantic_id = inst_id
                        else:
                            semantic_id = inst_id // 100
                        category_id = semantic_id - 1  # 与create_panoptic_video_labels.py第68行一致
                        category_name = self.CLASSES[category_id]
                        self.dataset_list.append({
                            'instance_id': inst_id,
                            'major_class': category_name,
                            'img_w': int(img_w),
                            'img_h': int(img_h),
                            'mask_files': mask_files_slice,
                            'img_files': img_files_slice
                        })
            print("data prepared")
            print("total dataset length: ", len(self.dataset_list))
            print("save prepared data to json file")
            with open(os.path.join(self.vipseg_dataset_path, "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json"), "w") as f:
                json.dump(self.dataset_list, f)
            print("data saved to json file")

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        seq_name = self.dataset_list[index]['major_class']
        mask_files = self.dataset_list[index]['mask_files']
        img_files = self.dataset_list[index]['img_files']
        # for i in range(len(mask_files)):
        #     assert os.path.basename(mask_files[i]).split(".")[0] == os.path.basename(img_files[i]).split(".")[0]
        #load the mask and img
        mask = []
        for i in range(len(mask_files)):
            if mask_files[i] is not None:
                mask.append(Image.open(mask_files[i]))
                mask[i] = mask[i].resize(self.target_resolution, Image.Resampling.NEAREST)
                mask[i] = np.array(mask[i])
                mask[i] = mask[i]==self.dataset_list[index]['instance_id']
            else:
                mask.append(None)
        img = [Image.open(img_file) for img_file in img_files]
        #resize the mask and img to the target resolution


        img = [img.resize(self.target_resolution, Image.Resampling.BICUBIC) for img in img]
        #convert the mask and img to numpy array
        img = [np.array(img) for img in img]
       

        return {
            'class': seq_name,
            'class_index': self.CLASSES_INDEX[seq_name],
            'mask': mask,
            'img': np.array(img),
            'instance_id': self.dataset_list[index]['instance_id']
        }

if __name__ == "__main__":
    dataset = VIPSeg_origin_mix_dataloader()
    print(len(dataset))
    count = 0
    for i in  tqdm( range(len(dataset))):

        data = dataset[i]
        print(data['class'])
        # print(data['mask'].shape)
        print(data['img'].shape)
        #save the masked img to png files
        for j in range(len(data['mask'])):
            if data['mask'][j] is not None:
                masked_img = (np.expand_dims(data['mask'][j],2) * data['img'][j])
                masked_img = Image.fromarray(masked_img)
                os.makedirs(os.path.join("/inspurfs/group/mayuexin/xiaqi/VIPSeg_origin_mix",data['class'],str(data['instance_id'])), exist_ok=True)
                masked_img.save(os.path.join("/inspurfs/group/mayuexin/xiaqi/VIPSeg_origin_mix",data['class'],str(data['instance_id']), str(j) + ".png"))
        count += 1
        if count > 30:
            break