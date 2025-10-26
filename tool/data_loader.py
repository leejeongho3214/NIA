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


class CustomDataset(Dataset):
    def __init__(self, args, logger, mode):
        self.args = args
        self.mode = mode
        self.logger = logger
        self.img_path = "dataset/img"
        self.json_path = "dataset/label"
        self.dataset_dict = defaultdict(list)
        self.grade_num = defaultdict(lambda: defaultdict(int))

        self.loading()

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return idx_list[idx], self.sub_path[idx_list[idx]], self.train_num

    def loading(self):
        if len(self.args.equ) == 1:
            if self.args.equ[0] == 1:
                device = "digital_camera"
            elif self.args.equ[0] == 2:
                device = "smart_pad"
            else:
                device = "smart_phone"

            with open(
                f"dataset/split/{self.args.mode}/{device}/{self.args.seed}_{self.mode}set_info.json",
                "r",
            ) as f:
                dataset_list = json.load(f)

        else:
            if self.mode == "train":
                dataset_list = defaultdict(list)
                for equ in self.args.equ:
                    if equ == 1:
                        device = "digital_camera"
                    elif equ == 2:
                        device = "smart_pad"
                    else:
                        device = "smart_phone"

                    with open(
                        f"dataset/split/{self.args.mode}/{device}/{self.args.seed}_{self.mode}set_info.json",
                        "r",
                    ) as f:
                        partial = json.load(f)

                    for class_name, files in partial.items():
                        dataset_list[class_name].extend(files)
                dataset_list = dict(dataset_list)
            else:
                equ = self.args.equ[-1]
                if equ == 1:
                    device = "digital_camera"
                elif equ == 2:
                    device = "smart_pad"
                else:
                    device = "smart_phone"
                with open(
                    f"dataset/split/{self.args.mode}/{device}/{self.args.seed}_{self.mode}set_info.json",
                    "r",
                ) as f:
                    dataset_list = json.load(f)

        for class_name, file_list in dataset_list.items():
            for file_name in file_list:
                sub, equ, angle, area = [file_name.split("_")[i] for i in [1, 3, 5, 7]]

                with open(
                    f"{self.json_path}/{equ}/{sub}/{sub}_{equ}_{angle}_{area}.json", "r"
                ) as f:
                    json_value = json.load(f)

                if self.args.mode == "class":
                    for full_name, value in json_value["annotations"].items():
                        if class_name in full_name:
                            self.dataset_dict[class_name].append(
                                [f"{equ}/{sub}/{sub}_{equ}_{angle}_{area}", value]
                            )
                            self.grade_num[class_name][value] += 1
                else:
                    for full_name, value in json_value["equipment"].items():
                        if class_name in full_name:
                            self.dataset_dict[class_name].append(
                                [f"{equ}/{sub}/{sub}_{equ}_{angle}_{area}", value]
                            )

    def save_dict(self, transform):
        ori_img = cv2.imread(os.path.join("dataset/cropped_img", self.i_path + ".jpg"))
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

    def load_dataset(self, dig):
        if self.args.mode == "class":
            grade_num = [
                self.grade_num[dig][key] for key in sorted(self.grade_num[dig].keys())
            ]
        self.area_list = list()
        self.dig = dig

        transform = transforms.Compose(
            [
                transforms.Resize((self.args.res, self.args.res), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        for self.i_path, self.grade in tqdm(self.dataset_dict[dig], desc=f"{self.dig}"):
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
