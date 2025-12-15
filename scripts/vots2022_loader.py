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

split_names = ["train", "val"]
class VOTS2022_dataloader:
    def __init__(self, dataset_path: str="/inspurfs/group/mayuexin/yaoych/LP/Dataset/vots2022-davis-format", split_name: str="train", 
    length: int=84, target_resolution: tuple[int, int]=(480, 270), 
    min_instance_ratio: float=0.10, min_bbox_ratio: float=0.10):
        self.dataset_path = dataset_path
        self.split_name = split_name
        assert split_name in split_names, "split_name must be in " + str(split_names)
        self.length = length
        self.target_resolution = target_resolution
        self.min_instance_ratio = min_instance_ratio
        self.min_bbox_ratio = min_bbox_ratio
        self.CLASSES = glob.glob(os.path.join(dataset_path, "Annotations", "Full-Resolution", "*"))
        print(self.CLASSES)
        self.CLASSES = [os.path.basename(cls).split("_")[0] for cls in self.CLASSES]
        print(self.CLASSES)

        #load from json file if exists
        if os.path.exists(os.path.join("/inspurfs/group/mayuexin/xiaqi/vots2022", "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json")):
            self.dataset_list = json.load(open(os.path.join("/inspurfs/group/mayuexin/xiaqi/vots2022", "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json"), "r"))
            print("data loaded from json file")
            print("total dataset length: ", len(self.dataset_list))
            return
        else:
            print("no json file found")

            if split_name == "val":
                self.dataset_list = []
                return 
            split_folder_names = glob.glob(os.path.join(dataset_path, "Annotations", "Full-Resolution", "*"))
            split_folder_names = [os.path.basename(cls) for cls in split_folder_names]
            self.split_folder_names = split_folder_names

            print("preparing data...")
            seq_iter = []
            for split_folder_name in self.split_folder_names:
                seq_name = split_folder_name.split("/")[-1]
                mask_files = glob.glob(os.path.join(dataset_path, "Annotations", "Full-Resolution", seq_name, "*.png"))
                img_files = glob.glob(os.path.join(dataset_path, "JPEGImages", "Full-Resolution", seq_name, "*.jpg"))
                mask_files.sort()
                img_files.sort()
                if len(img_files) < length:
                    continue
                seq_iter.append((seq_name, mask_files, img_files))
            seq_iter =tqdm(seq_iter, desc="preparing data", unit="seq") 
            self.dataset_list = []
            for seq_name, mask_files, img_files in seq_iter:
            # 图像尺寸（优先 JPEGImages）


                with Image.open(mask_files[0]) as m0:
                    img_w, img_h = m0.size

                instance_ratios_dict: Dict[int, List[float]] = {}  # inst_id -> ratios list
                bbox_ratios_dict: Dict[int, List[float]] = {} 
                frame_iter = tqdm(mask_files, desc=f'{seq_name} masks', unit='frame', leave=False)

                for mf in frame_iter:
                    with Image.open(mf) as m:
                        arr = np.array(m)
                        if arr.ndim == 3:
                            arr = arr[..., 0]
                        # 找到所有实例 id（>0）
                        ids = np.unique(arr)
                        ids = ids[ids > 0]
                        if ids.size == 0:
                            continue
                        
                        
                        if img_w <= 0 or img_h <= 0:
                            h, w = arr.shape[:2]
                            denom = float(w * h)
                        else:
                            denom = float(img_w * img_h)
                        if denom <= 0:
                            continue
                        
                        for inst_id in ids.tolist():
                            mask = (arr == inst_id)
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
                        for i in range(len(mask_files)-self.length+1):
                            with Image.open(mask_files[i]) as m:
                                arr = np.array(m)
                                if arr.ndim == 3:
                                    arr = arr[..., 0]
                                if inst_id not in np.unique(arr):
                                    continue
                            mask_files_slice = mask_files[i:i+self.length]
                            img_files_slice = img_files[i:i+self.length]
                            self.dataset_list.append({
                                'instance_id': inst_id,
                                'major_class': seq_name.split("_")[0],  # DAVIS17使用序列名作为major_class
                                'img_w': int(img_w),
                                'img_h': int(img_h),
                                'mask_files': mask_files_slice,
                                'img_files': img_files_slice
                            })
            print("data prepared")
            print("total dataset length: ", len(self.dataset_list))
            print("save prepared data to json file")
            with open(os.path.join("/inspurfs/group/mayuexin/xiaqi/vots2022", "dataset_list_" + self.split_name + "_" + str(self.length)+ "_" + str(round( self.min_instance_ratio*100, 2)) + "_" + str(round( self.min_bbox_ratio*100, 2)) + ".json"), "w") as f:
                json.dump(self.dataset_list, f)
            print("data saved to json file")

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        seq_name = self.dataset_list[index]['major_class']
        mask_files = self.dataset_list[index]['mask_files']
        img_files = self.dataset_list[index]['img_files']
        for i in range(len(mask_files)):
            assert os.path.basename(mask_files[i]).split(".")[0] == os.path.basename(img_files[i]).split(".")[0]
        #load the mask and img
        mask = [Image.open(mf) for mf in mask_files]
        img = [Image.open(img_file) for img_file in img_files]
        #resize the mask and img to the target resolution
        mask = [mf.resize(self.target_resolution, Image.Resampling.NEAREST) for mf in mask]


        img = [img.resize(self.target_resolution, Image.Resampling.BICUBIC) for img in img]
        #convert the mask and img to numpy array
        mask = [np.array(mf) for mf in mask]
        mask = [mf==self.dataset_list[index]['instance_id'] for mf in mask]
        img = [np.array(img) for img in img]
       

        return {
            'class': seq_name,
            'class_index': self.CLASSES.index(seq_name),
            'mask': np.array(mask),
            'img': np.array(img)
        }

if __name__ == "__main__":
    dataset = VOTS2022_dataloader(dataset_path="/inspurfs/group/mayuexin/yaoych/LP/Dataset/vots2022-davis-format", 
        split_name="train", 
        length=81, 
        target_resolution=(480, 270), 
        min_instance_ratio=0.1, 
        min_bbox_ratio=0.1)
    print(len(dataset))
    for i in  tqdm( range(len(dataset))):
        if dataset.dataset_list[i]['major_class'] == "singer":
            data = dataset[0]
            print(data['class'])
            print(data['mask'].shape)
            print(data['img'].shape)
            #save the masked img to png files
            for j in range(len(data['mask'])):
                masked_img = (np.expand_dims(data['mask'][j],2) * data['img'][j])
                masked_img = Image.fromarray(masked_img)
                os.makedirs(os.path.join("/inspurfs/group/mayuexin/xiaqi/vots2022",data['class']), exist_ok=True)
                masked_img.save(os.path.join("/inspurfs/group/mayuexin/xiaqi/vots2022",data['class'], str(j) + ".png"))
            break