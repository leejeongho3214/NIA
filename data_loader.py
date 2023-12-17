from PIL import Image
import natsort
import torch
from torchvision import transforms
import os
import numpy as np
import cv2
import json
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import Dataset

folder_name = {
    "F": "01",
    "Fb": "07",
    "Ft": "06",
    "L15": "02",
    "L30": "03",
    "R15": "04",
    "R30": "05",
    "L": "02",
    "R": "03",
}

class_num_list = {
    "pigmentation": 6,
    "wrinkle": 7,
    "pore": 6,
    "dryness": 5,
    "sagging": 7,
}


area_naming = {
    "0": "all",
    "1": "forehead",
    "2": "glabellus",
    "3": "l_peroucular",
    "4": "r_peroucular",
    "5": "l_cheek",
    "6": "r_cheek",
    "7": "lip",
    "8": "chin",
}

img_num = {
    "01": 7,
    "02": 3,
    "03": 3,
}


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_list(args)
        self.train_list, self.val_list, self.test_list = random_split(
            self.dataset, [0.8, 0.1, 0.1]
        )
        
    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        return self.sub_path[idx]

    def load_list(self, args):
        self.img_path = args.img_path
        self.dataset = list()
        self.json_path = args.json_path
        sub_path_list = [
            item
            for item in natsort.natsorted(os.listdir(self.img_path))
            if not item.startswith(".")
        ]
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize(
                    [0.24628267, 0.3271797, 0.44643742],
                    [0.1666497, 0.2335198, 0.3375362],
                )]
        self.transform = transforms.Compose(transform_list)

        for equ_name in sub_path_list:
            if equ_name.startswith(".") or int(equ_name) not in self.args.equ:
                continue

            for sub_fold in natsort.natsorted(
                os.listdir(os.path.join(self.img_path, equ_name))
            ):
                if sub_fold.startswith(".") or not os.path.exists(
                    os.path.join(self.json_path, equ_name, sub_fold)
                ):
                    continue

                folder_path = os.path.join(self.img_path, equ_name, sub_fold)
                for img_name in natsort.natsorted(os.listdir(folder_path)):
                    if not img_name.endswith((".png", ".jpg", ".jpeg")):
                        continue
                    
                    if img_name.split('.')[0].split('_')[-1] != "F":
                        continue
                    
                    pre_name = '/'.join(folder_path.split('/')[2:])
                    json_name = os.path.join('dataset/label', pre_name)

                    self.dataset.append(
                        {
                            "equ_name": equ_name,
                            "folder_path": folder_path,
                            "img_name": img_name,
                        }
                    )

        self.dataset = self.dataset[: self.args.data_num]

    def load_dataset(self, args, mode):
        self.sub_path = list()
        data_list = (
            self.train_list
            if mode == "train"
            else self.val_list
            if mode == "val"
            else self.test_list
        )
        for value in tqdm(data_list):
            equ_name = value["equ_name"]
            folder_path = value["folder_path"]
            sub_fold = folder_path.split("/")[-1]
            img_name = value["img_name"]

            angle = img_name.split(".")[0].split("_")[-1]
            img = cv2.imread(os.path.join(folder_path, img_name))
            area_list = dict()
            start_idx = 1 if args.mode == "class" else 0
            for idx_area in range(start_idx, 9):
                try:
                    (
                        area_name,
                        meta,
                        patch_img,
                    ) = self.load_img(img_name, angle, idx_area, equ_name, img, args)

                except:
                    continue

                pil_img = Image.fromarray(patch_img)
                patch_img = self.transform(pil_img)
                label_data = (
                    meta["annotations"] if args.mode == "class" else meta["equipment"]
                )

                if type(label_data) != dict:
                    continue

                if args.mode != "class":
                    label_data = self.norm_reg(meta, idx_area)

                desc_area = (
                    "Sub_"
                    + sub_fold
                    + "_Equ_"
                    + equ_name
                    + "_Angle_"
                    + angle
                    + "_Area_"
                    + area_name
                )

                for key in label_data:
                    dig = key.split("_")[-1]
                    if dig not in area_list:
                        area_list[dig] = list()
                    area_list[dig].append([patch_img, label_data[key], desc_area])

            self.sub_path.append(area_list)

    def load_img(self, img_name, angle, idx_area, equ_name, img, args):
        json_name = "_".join(img_name.split("_")[:2]) + f"_{angle}_{idx_area:02d}.json"
        with open(
            os.path.join(
                self.json_path,
                equ_name,
                json_name.split("_")[0],
                json_name,
            ),
            "r",
            encoding="utf8",
        ) as f:
            meta = json.load(f)

        if meta["images"]["bbox"] == None:
            return 1

        bbox_point = [int(item) for item in meta["images"]["bbox"]]
        bbox_x = [
            min(bbox_point[0], bbox_point[2]),
            max(bbox_point[0], bbox_point[2]),
        ]
        bbox_y = [
            min(bbox_point[1], bbox_point[3]),
            max(bbox_point[1], bbox_point[3]),
        ]

        area_name = str(int(json_name.split("_")[-1].split(".")[0]))
        patch_img = img[bbox_y[0] : bbox_y[1], bbox_x[0] : bbox_x[1]]
        height, width = patch_img.shape[0] // 32, patch_img.shape[1] // 32
        patch_img = cv2.resize(patch_img, (width * 32, height * 32) )

        return area_name, meta, patch_img

    def norm_reg(self, meta, idx_area):
        item_list = dict()
        type_class = {
            "0": ["pigmentation_count"],
            "1": ["forehead_moisture", "forehead_elasticity_R2"],
            "3": ["l_perocular_wrinkle_Ra"],
            "4": ["r_perocular_wrinkle_Ra"],
            "5": ["l_cheek_moisture", "l_cheek_elasticity_R2", "l_cheek_pore"],
            "6": ["r_cheek_moisture", "r_cheek_elasticity_R2", "r_cheek_pore"],
            "8": ["chin_moisture", "chin_elasticity_R2"],
        }

        for item in type_class[f"{idx_area}"]:
            split_list = item.split("_")
            dig_v = split_list[-1]
            # if meta["equipment"][item] == "Er":
            #     item_list.append(np.nan)
            dig = split_list[-1]
            if dig in ["Ra", "R2"]:
                dig = split_list[-2]

            if dig_v == "R2":
                item_list[dig] = meta["equipment"][item]

            elif dig_v == "moisture":
                item_list[dig] = meta["equipment"][item] / 100

            elif dig_v == "Ra":
                item_list[dig] = meta["equipment"][item] / 50

            elif dig_v == "count":
                item_list[dig] = meta["equipment"][item] / 350

            elif dig_v == "pore":
                item_list[dig] = meta["equipment"][item] / 2600

            else:
                assert 0, "dig_v is not here"

            item_list[dig] = round(item_list[dig], 5)

        return item_list
