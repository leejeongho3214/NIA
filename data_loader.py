import errno
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
from torch.utils.data import random_split, ConcatDataset, Dataset


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


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
        train_list, val_list, test_list = list(), list(), list()

        for dig, class_dict in self.json_dict.items():
            for grade, name_list in class_dict.items():
                if len(name_list) < 10:
                    name_list = name_list * (10 // len(name_list) + 1)
                t, v, tt = random_split(name_list, [0.8, 0.1, 0.1])
                train_list.append([dig, grade, t])
                val_list.append([dig, grade, v])
                test_list.append([dig, grade, tt])

        self.train_list, self.val_list, self.test_list = (
            ConcatDataset(train_list),
            ConcatDataset(val_list),
            ConcatDataset(test_list),
        )

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return idx_list[idx], self.sub_path[idx_list[idx]], self.train_num

    def load_list(self, args):
        self.img_path = args.img_path
        self.json_path = args.json_path
        sub_path_list = [
            item
            for item in natsort.natsorted(os.listdir(self.img_path))
            if not item.startswith(".")
        ]
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.24628267, 0.3271797, 0.44643742],
                [0.1666497, 0.2335198, 0.3375362],
            ),
        ])
        
        self.transform_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(200),
            transforms.Resize(256),
            transforms.Normalize(
                [0.24628267, 0.3271797, 0.44643742],
                [0.1666497, 0.2335198, 0.3375362],
            ),
        ])
        
        self.transform_crop1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(210),
            transforms.Resize(256),
            transforms.Normalize(
                [0.24628267, 0.3271797, 0.44643742],
                [0.1666497, 0.2335198, 0.3375362],
            ),
        ])
        
        self.transform_color = transforms.Compose([
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Normalize(
                [0.24628267, 0.3271797, 0.44643742],
                [0.1666497, 0.2335198, 0.3375362],
            ),
        ])
        
        self.transform_both = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(200),
            transforms.Resize(256),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.Normalize(
                [0.24628267, 0.3271797, 0.44643742],
                [0.1666497, 0.2335198, 0.3375362],
            ),
        ])
        
        self.json_dict = defaultdict(lambda: defaultdict(list))
        for equ_name in sub_path_list:
            if equ_name.startswith(".") or int(equ_name) not in self.args.equ:
                continue

            for sub_fold in tqdm(
                natsort.natsorted(os.listdir(os.path.join(self.img_path, equ_name))),
                desc="path loading..",
            ):
                if sub_fold.startswith(".") or not os.path.exists(
                    os.path.join(self.json_path, equ_name, sub_fold)
                ):
                    continue

                folder_path = os.path.join(self.img_path, equ_name, sub_fold)
                for img_name in natsort.natsorted(os.listdir(folder_path)):
                    if not img_name.endswith((".png", ".jpg", ".jpeg")):
                        continue
                    
                    if img_name.split(".")[0].split("_")[-1] != "F":
                        continue
                    
                    pre_name = "/".join(folder_path.split("/")[2:])
                    json_name = os.path.join("dataset/label", pre_name)
                    img_name = os.path.join("dataset/img", pre_name)

                    for j_name in os.listdir(json_name):
                        if j_name.split("_")[2] == "F":
                            with open(os.path.join(json_name, j_name), "r") as f:
                                json_meta = json.load(f)
                                if list(json_meta["annotations"].keys())[0] == "acne":
                                    continue

                                for dig_n, grade in json_meta["annotations"].items():
                                    dig, area = (
                                        dig_n.split("_")[-1],
                                        dig_n.split("_")[-2],
                                    )
                                    if dig in [
                                        "wrinkle",
                                        "pigmentation",
                                    ]:  ## Only one area per each class
                                        if area != "forehead":
                                            continue
                                    self.json_dict[dig][str(grade)].append(
                                        os.path.join(pre_name, j_name.split(".")[0])
                                    )

    def load_dataset(self, args, mode):
        self.sub_path = defaultdict(list)
        self.train_num = defaultdict(lambda: defaultdict(int))
        data_list = (
            self.train_list
            if mode == "train"
            else self.val_list
            if mode == "val"
            else self.test_list
        )
        area_list = defaultdict(lambda: defaultdict(list))
        
        def save_dict(transform):
            pil = Image.fromarray(pil_img.astype(np.uint8))
            patch_img = transform(pil)

            with open(os.path.join("dataset/label", i_path + ".json"), "r") as f:
                meta = json.load(f)

            label_data = (
                meta["annotations"] if args.mode == "class" else meta["equipment"]
            )

            for key in label_data:
                dig_p = key.split("_")[-1]
                if dig == dig_p:
                    area_list[dig][str(label_data[key])].append(
                        [patch_img, label_data[key], desc_area, dig]
                    )

        for dig, grade, class_dict in tqdm(data_list.datasets, desc=f"{mode}_class"): 
            for idx, i_path in enumerate(
                tqdm(sorted(class_dict), desc=f"{dig}_{grade}")
            ):
                if idx == self.args.data_num:
                    break
                p_img = cv2.imread(os.path.join("dataset/cropped_img", i_path + ".jpg"))
                r_value = 256 / max(p_img.shape)

                pil_img = np.zeros([256, 256, 3])
                r_img = cv2.resize(
                    p_img,
                    (int(p_img.shape[1] * r_value), int(p_img.shape[0] * r_value)),
                )
                pil_img[: r_img.shape[0], : r_img.shape[1]] = r_img
                if i_path.split("_")[-1] in ["04", "06"]:
                    pil_img = cv2.flip(pil_img, 1)
                
                s_list = i_path.split("/")[-1].split("_")
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
                
                if mode == "train":
                    for transform in [self.transform_test]:
                        if s_list[3] in ['01', '02', '07', '08']:
                            save_dict(transform)
                            pil_img = cv2.flip(pil_img, 1)
                            save_dict(transform)
                        else:
                            save_dict(transform)
                else:
                    save_dict(self.transform_test)
                    

        for k, v in area_list.items():
            self.sub_path[k] = [item for items in v.values() for item in items]
            for items in v.values():
                self.train_num[items[0][-1]][str(items[0][1])] = len(items)

        return self.sub_path, self.train_num

        # for dig, class_dict in area_list.items():
        #     grade_list = [
        #         len(total_list) for _, total_list in sorted((class_dict.items()))
        #     ]
        #     guide_num = max(grade_list)

        #     for grade, total_list in sorted(class_dict.items()):
        #         mul_num, remain_num = guide_num // len(total_list), guide_num % len(
        #             total_list
        #         )
        #         sub_path[dig].append(total_list * mul_num + total_list[:remain_num])

        # for k, v in sub_path.items():
        #     self.sub_path[k] = [item for items in v for item in items]
        # del sub_path, area_list

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

        reduction_value = max(patch_img.shape) / args.res

        age = torch.tensor([round((meta["info"]["age"] - 13.0) / 69.0, 5)])
        gender = 0 if meta["info"]["gender"] == "F" else 1
        gender_l = F.one_hot(torch.tensor(gender), 2)

        meta_v = torch.concat([age, gender_l])

        return reduction_value, area_name, meta, patch_img, meta_v

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
