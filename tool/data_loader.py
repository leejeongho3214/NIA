import copy
import errno
import random
from PIL import Image
import natsort
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import cv2
import json
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset


def mkdir(path):
    if path == "":
        return
    try:    
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

class CustomDataset_class(Dataset):
    def __init__(self, args, logger, mode):
        self.args = args
        self.logger = logger

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return idx_list[idx], self.sub_path[idx_list[idx]], self.train_num

    def process_json_meta(
        self, json_meta, j_name, sub_path, target_list, sub_fold
    ):
        if (
            (
                list(json_meta["annotations"].keys())[0] == "acne"
                or json_meta["images"]["bbox"] is None
            )
            and self.args.mode == "class"
        ) or (
            (json_meta["equipment"] is None or json_meta["images"]["bbox"] is None)
            and self.args.mode == "regression"
        ):
            return

        age = torch.tensor([round((json_meta["info"]["age"] - 13.0) / 69.0, 5)])
        gender = 0 if json_meta["info"]["gender"] == "F" else 1
        gender_l = F.one_hot(torch.tensor(gender), 2)
        meta_v = torch.cat([age, gender_l])

        if self.args.mode == "class":
            for dig_n, grade in json_meta["annotations"].items():
                if (dig_n == "chin_sagging" and grade == 6):
                    continue
                
                dig, area = dig_n.split("_")[-1], dig_n.split("_")[-2]

                if dig in ["wrinkle", "pigmentation"] and self.mode == "test":
                    dig = f"{dig}_{area}"

                self.json_dict[dig][str(grade)][sub_fold].append(
                    [os.path.join(sub_path, j_name.split(".")[0]), meta_v]
                )
        else:
            for dig_n, value in json_meta["equipment"].items():
                matching_dig = [
                    dig_n for target_dig in target_list if target_dig in dig_n
                ]
                if matching_dig:
                    dig = matching_dig[0]
                    self.json_dict[dig][value][sub_fold].append(
                        [
                            os.path.join(sub_path, j_name.split(".")[0]),
                            value,
                            meta_v,
                        ]
                    )

    def save_dict(self, transform):
        ori_img = cv2.imread(os.path.join("dataset/cropped_img", self.i_path + ".jpg"))     ## Be careful the path of the dataset whether it is 0-padding or not
        pil_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)  
        ori_img = cv2.resize(ori_img, (self.args.res, self.args.res))

        s_list = self.i_path.split("/")[-1].split("_")
        desc_area = (
            "Sub_"
            + s_list[0]
            + "_Equ_"
            + s_list[1]
            + "_Angle_"
            + s_list[2]
            + "_Area_"
            + s_list[3]
        )

        if self.args.mode == "class":
            label_data = int(self.grade)
        else:
            norm_v = self.norm_reg(self.grade)
            label_data = norm_v

        pil = Image.fromarray(pil_img.astype(np.uint8))
        patch_img = transform(pil)

        self.area_list.append([patch_img, label_data, desc_area, self.dig, 0, ori_img])

                            
                            
    def load_dataset(self, mode, dig):
        self.mode = mode
        self.img_path = "dataset/img"
        self.json_path = "dataset/label"

        if self.args.equ == 1:
            device = "digital_camera"
        elif self.args.equ == 2:
            device = "smart_pad"
        else:
            device = "smart_phone"
            
        with open(f"dataset/split/{device}/{self.args.seed}_{mode}set_info.txt", "r") as f:
            dataset_list = json.load(f)
        
        self.dataset_dict = defaultdict(list)
        self.grade_num = defaultdict(lambda: defaultdict(int))
        for class_name, file_list in dataset_list.items():
            for file_name in file_list:            
                sub, equ, angle, area = [file_name.split("_")[i] for i in [1, 3, 5, 7]]
                
                with open(f"{self.json_path}/{equ}/{sub}/{sub}_{equ}_{angle}_{area}.json", "r") as f:
                    json_value = json.load(f)
                
                if self.args.mode == "class":
                    for class_item, value in json_value["annotations"].items():
                        if class_name in class_item:
                            self.dataset_dict[class_name].append([f"{equ}/{sub}/{sub}_{equ}_{angle}_{area}", value])
                            self.grade_num[class_name][value] += 1
                else:
                    for class_item, value in json_value["equipment"].items():
                        if class_name in class_item:
                            self.dataset_dict[class_name].append([f"{equ}/{sub}/{sub}_{equ}_{angle}_{area}", value])
        
        grade_num = [self.grade_num[dig][key] for key in sorted(self.grade_num[dig].keys())] 
        self.area_list = list()
        self.dig = dig

        transform = transforms.Compose(
            [
                transforms.Resize((256, 256), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        for self.i_path, self.grade in tqdm(self.dataset_dict[dig], desc = f"{self.dig}"):
            self.save_dict(transform)

        if self.args.mode == "class":
            return self.area_list, grade_num
        else:
            return self.area_list, 0 

    def norm_reg(self, value):
        dig_v = self.dig.split("_")[-1]

        if dig_v == "R2":
            return value

        elif dig_v == "moisture":
            return value / 100

        elif dig_v == "Ra":
            return value / 50

        elif dig_v in ["pigmentation", "count"]:
            return value / 350

        elif dig_v == "pore":
            return value / 2600

        else:
            assert 0, "dig_v is not here"

class CustomDataset_regress(CustomDataset_class):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

