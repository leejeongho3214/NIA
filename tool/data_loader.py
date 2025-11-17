import copy
import errno
import random
from PIL import Image
import natsort
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import os
import numpy as np
import cv2
import json
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset


AUGMENTATION_PRESETS = {
    "none": {
        "flip_prob": 0.0,
        "color_jitter": 0.0,
        "rotation": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "perspective_prob": 0.0,
        "blur_prob": 0.0,
        "sharp_prob": 0.0,
        "erase_prob": 0.0,
        "erase_scale": (0.0, 0.0),
    },
    "light": {
        "flip_prob": 0.4,
        "color_jitter": 0.15,
        "rotation": 5.0,
        "translate": 0.02,
        "scale": 0.05,
        "shear": 2.0,
        "perspective": 0.03,
        "perspective_prob": 0.15,
        "blur_prob": 0.1,
        "sharp_prob": 0.1,
        "erase_prob": 0.05,
        "erase_scale": (0.01, 0.06),
    },
    "medium": {
        "flip_prob": 0.5,
        "color_jitter": 0.25,
        "rotation": 8.0,
        "translate": 0.04,
        "scale": 0.08,
        "shear": 4.0,
        "perspective": 0.04,
        "perspective_prob": 0.2,
        "blur_prob": 0.15,
        "sharp_prob": 0.15,
        "erase_prob": 0.1,
        "erase_scale": (0.02, 0.12),
    },
    "heavy": {
        "flip_prob": 0.6,
        "color_jitter": 0.35,
        "rotation": 10.0,
        "translate": 0.05,
        "scale": 0.1,
        "shear": 6.0,
        "perspective": 0.06,
        "perspective_prob": 0.25,
        "blur_prob": 0.2,
        "sharp_prob": 0.2,
        "erase_prob": 0.15,
        "erase_scale": (0.03, 0.2),
    },
}


def mkdir(path):
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


class CustomDataset(Dataset):
    def __init__(self, args, logger, mode, special = False):
        self.args = args
        self.mode = mode
        self.logger = logger
        self.img_path = "dataset/img"
        self.json_path = "dataset/label"
        self.dataset_dict = defaultdict(list)
        self.grade_num = defaultdict(lambda: defaultdict(int))
        self.loading(special)
        self._train_transform = self._build_transform(train=True)
        self._eval_transform = self._build_transform(train=False)

    def _build_transform(self, train: bool):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        resize = transforms.Resize((self.args.res, self.args.res), antialias=True)
        aug_level = getattr(self.args, "aug_level", "none")
        config = AUGMENTATION_PRESETS.get(aug_level, AUGMENTATION_PRESETS["light"])

        ops = [resize]
        if train and aug_level != "none":
            if getattr(self.args, "allow_hflip", False) and config["flip_prob"] > 0:
                ops.append(transforms.RandomHorizontalFlip(p=config["flip_prob"]))

            if any(
                [
                    config["rotation"] > 0,
                    config["translate"] > 0,
                    config["scale"] > 0,
                    config["shear"] > 0,
                ]
            ):
                translate = (
                    (config["translate"], config["translate"])
                    if config["translate"] > 0
                    else None
                )
                scale = (
                    (max(0.5, 1.0 - config["scale"]), 1.0 + config["scale"])
                    if config["scale"] > 0
                    else None
                )
                shear = (
                    (-config["shear"], config["shear"])
                    if config["shear"] > 0
                    else None
                )
                ops.append(
                    transforms.RandomAffine(
                        degrees=(
                            (-config["rotation"], config["rotation"])
                            if config["rotation"] > 0
                            else 0
                        ),
                        translate=translate,
                        scale=scale,
                        shear=shear,
                        interpolation=InterpolationMode.BILINEAR,
                        fill=0,
                    )
                )

            if config["color_jitter"] > 0:
                jitter = config["color_jitter"]
                ops.append(
                    transforms.ColorJitter(
                        brightness=jitter,
                        contrast=jitter,
                        saturation=min(jitter * 1.5, 0.75),
                        hue=min(jitter * 0.2, 0.15),
                    )
                )

            if config["perspective_prob"] > 0 and config["perspective"] > 0:
                ops.append(
                    transforms.RandomPerspective(
                        distortion_scale=config["perspective"],
                        p=config["perspective_prob"],
                    )
                )

            if config["blur_prob"] > 0:
                kernel = 5 if self.args.res >= 384 else 3
                ops.append(
                    transforms.RandomApply(
                        [transforms.GaussianBlur(kernel_size=kernel, sigma=(0.1, 1.5))],
                        p=config["blur_prob"],
                    )
                )

            if config["sharp_prob"] > 0:
                ops.append(
                    transforms.RandomAdjustSharpness(
                        sharpness_factor=1.2, p=config["sharp_prob"]
                    )
                )

        tensor_ops = [transforms.ToTensor()]
        if train and config["erase_prob"] > 0:
            tensor_ops.append(
                transforms.RandomErasing(
                    p=config["erase_prob"],
                    scale=config["erase_scale"],
                    value="random",
                )
            )
        tensor_ops.append(normalize)

        return transforms.Compose(ops + tensor_ops)

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return idx_list[idx], self.sub_path[idx_list[idx]], self.train_num
    
    def get_device_name(self, equ):
        if equ == 1:
            device = "digital_camera"
        elif equ == 2:
            device = "smart_pad"
        else:
            device = "smart_phone"
            
        return device

    def _dataset_info_path(self, device, special):
        split = "split3" if special else "split2"
        return f"dataset/{split}/{self.args.mode}/{device}/{self.args.seed}_{self.mode}set_info.json"

    def _load_device_dataset(self, equ, special):
        device = self.get_device_name(equ)
        with open(self._dataset_info_path(device, special), "r") as f:
            return json.load(f)

    def _merge_datasets(self, equ_list, special):
        merged = defaultdict(list)
        for equ in equ_list:
            partial = self._load_device_dataset(equ, special)
            for class_name, files in partial.items():
                merged[class_name].extend(files)
        return dict(merged)

    def loading(self, special):
        if len(self.args.equ) == 1:
            dataset_list = self._load_device_dataset(self.args.equ[0], special)
        else:
            dataset_list = self._merge_datasets(self.args.equ, special)

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

        transform = (
            self._train_transform if self.mode == "train" else self._eval_transform
        )

        for self.i_path, self.grade in tqdm(self.dataset_dict[dig] , desc=f"{self.dig}"):
            if not os.path.isfile(os.path.join("dataset/cropped_img", self.i_path + ".jpg")):
                continue
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
