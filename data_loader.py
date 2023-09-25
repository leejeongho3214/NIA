from PIL import Image
import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt

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
    "wrinkle": 9,
    "pore": 6,
    "dryness": 5,
    "sagging": 7,
}


area_naming = {
    "0": "forearm",
    "1": "glabellus",
    "2": "l_peroucular",
    "3": "r_peroucular",
    "4": "l_cheek",
    "5": "r_cheek",
    "6": "lip",
    "7": "chin",
}

area_name = {
    "0": "forearm",
    "1": "glabellus",
    "2": "peroucular",
    "3": "_",
    "4": "cheek",
    "5": "_",
    "6": "lip",
    "7": "chin",
}

regression_name = {}


class CustomDataset(Dataset):
    def __init__(self, args):
        self.img_path = args.img_path
        self.json_path = args.json_path
        sub_path_list = os.listdir(self.img_path)
        sub_path_list = [f"{equ:02}" for equ in args.equ]
        element = (
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
            if args.normalize
            else [
                transforms.ToTensor(),
            ]
        )

        self.transform = {
            "train": transforms.Compose(element),
            "test": transforms.Compose(element),
        }
        self.sub_path = list()
        for equ_name in sub_path_list:
            for sub_fold in os.listdir(os.path.join(self.img_path, equ_name)):
                if sub_fold.startswith("."):
                    continue
                img_count = 0
                for file_name in os.listdir(
                    os.path.join(self.img_path, equ_name, sub_fold)
                ):
                    if any(
                        file_name.lower().endswith(ext) for ext in [".jpg", ".jpeg"]
                    ):
                        img_count += 1
                if img_count == 7:
                    folder_path = os.path.join(self.img_path, equ_name, sub_fold)
                    for img_name in os.listdir(folder_path):
                        if not img_name.endswith((".jpg", ".jpeg")):  ## 일부러 F를 추가함
                            continue

                        if args.angle == "F":
                            if not img_name.endswith(("F.jpg")):
                                continue
                        angle = img_name.split(".")[0].split("_")[-1]

                        img = cv2.imread(os.path.join(folder_path, img_name))
                        area_list = dict()
                        for idx_area in range(1, 9):
                            (
                                reduction_value,
                                bbox_x,
                                json_name,
                                area_name,
                                meta,
                                patch_img,
                            ) = self.load_img(
                                img_name, angle, idx_area, equ_name, img, args
                            )
                            try:
                                n_patch_img = cv2.resize(
                                    patch_img,
                                    (
                                        int(patch_img.shape[1] / reduction_value),
                                        int(patch_img.shape[0] / reduction_value),
                                    ),
                                )
                            except:
                                print(
                                    f"Sub No: {json_name.split('_')[0]} & Angle: {angle} & Area: {area_naming[area_name]} , w: {bbox_x[1]- bbox_x[0]}"
                                )
                                continue

                            # if args.double:
                            patch_img = self.make_double(idx_area, args, n_patch_img)

                            # else:
                            #     patch_img = np.zeros(
                            #         [args.res, args.res, 3], dtype=np.uint8
                            #     )
                            #     patch_img[
                            #         : n_patch_img.shape[0], : n_patch_img.shape[1]] = n_patch_img

                            pil_img = Image.fromarray(patch_img)
                            patch_img = self.transform["train"](pil_img)
                            label_data = (
                                meta["annotations"]
                                if args.mode == "class"
                                else meta["equipment"]
                            )
                            if label_data == None:
                                continue

                            if args.mode != "class":
                                item_list = self.norm_reg(meta)

                                label_data = torch.tensor(item_list)

                            area_list[f"{idx_area}"] = [patch_img, label_data]
                        
                        self.sub_path.append(area_list)

                else:
                    print(
                        f"\033[94m{os.path.join(self.img_path, equ_name, sub_fold)}는 제외됨\033[0m"
                    )

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        return self.sub_path[idx]

    def make_double(self, idx_area, args, n_patch_img):
        if idx_area in [1, 7, 8]:
            patch_img = np.zeros((128, 256, 3), dtype=np.uint8)
            if n_patch_img.shape[0] * 2 > 128:
                reduction_value = 64 / n_patch_img.shape[0]
                patch = cv2.resize(
                    n_patch_img,
                    (
                        int(128 * 2 * reduction_value),
                        int(n_patch_img.shape[0] * 2 * reduction_value),
                    ),
                )
                patch_img[:, : int(128 * 2 * reduction_value)] = patch
            else:
                patch = cv2.resize(
                    n_patch_img,
                    (128 * 2, n_patch_img.shape[0] * 2),
                )
                patch_img[: n_patch_img.shape[0] * 2] = patch

        elif idx_area in [3, 4]:
            patch_img = np.zeros((256, 128, 3), dtype=np.uint8)

            if n_patch_img.shape[1] * 2 > 128:
                reduction_value = 64 / n_patch_img.shape[1]
                patch = cv2.resize(
                    n_patch_img,
                    (
                        int(n_patch_img.shape[1] * 2 * reduction_value),
                        int(128 * 2 * reduction_value),
                    ),
                )
                patch_img[: int(128 * 2 * reduction_value)] = patch

            else:
                patch = cv2.resize(
                    n_patch_img,
                    (
                        n_patch_img.shape[1] * 2,
                        128 * 2,
                    ),
                )
                patch_img[:, : n_patch_img.shape[1] * 2] = patch

        else:
            patch_img = np.zeros((128, 128, 3), dtype=np.uint8)
            patch_img[: n_patch_img.shape[0], : n_patch_img.shape[1]] = n_patch_img

        return patch_img

    def load_img(self, img_name, angle, idx_area, equ_name, img, args):
        json_name = "_".join(img_name.split("_")[:2]) + f"_{angle}_{idx_area:02d}.json"
        with open(
            os.path.join(
                self.json_path,
                equ_name,
                folder_name[angle],
                json_name.split("_")[0],
                json_name,
            ),
            "r",
            encoding="utf8",
        ) as f:
            meta = json.load(f)

        bbox_point = meta["images"]["bbox"]
        bbox_x = [
            min(bbox_point[0], bbox_point[2]),
            max(bbox_point[0], bbox_point[2]),
        ]
        bbox_y = [
            min(bbox_point[1], bbox_point[3]),
            max(bbox_point[1], bbox_point[3]),
        ]
        area_name = str(int(json_name.split("_")[-1].split(".")[0]))
        if (bbox_point[0] > bbox_point[2]) or (bbox_point[1] > bbox_point[3]):
            print(
                f"Sub No: {json_name.split('_')[0]} & Angle: {angle} & Area: {area_naming[area_name]}// 위치 반전"
            )

        patch_img = img[bbox_y[0] : bbox_y[1], bbox_x[0] : bbox_x[1]]

        reduction_value = max(patch_img.shape) / args.res

        return reduction_value, bbox_x, json_name, area_name, meta, patch_img

    def norm_reg(self, meta):
        item_list = list()
        for item in meta["equipment"]:
            if type(meta["equipment"][item]) == dict:
                if item == "elasticity":
                    meta["equipment"]["elasticity"] = {
                        "R2": meta["equipment"][item]["R2"]
                    }

                if item == "wrinkle":
                    meta["equipment"]["wrinkle"] = {"Ra": meta["equipment"][item]["Ra"]}

                for items in meta["equipment"][item]:
                    if not type(meta["equipment"][item][items]) in [float, int]:
                        meta["equipment"][item][items] = np.nan

                    else:
                        if item == "wrinkle":
                            meta["equipment"][item][items] = (
                                meta["equipment"][item][items] / 100
                            )

                        elif item == "pigmentaion":
                            meta["equipment"][item][items] = (
                                meta["equipment"][item][items] / 300
                            )

                        elif item == "pore":
                            meta["equipment"][item][items] = (
                                meta["equipment"][item][items] / 3000
                            )

                    item_list.append(meta["equipment"][item][items])

            else:
                if not type(meta["equipment"][item]) in [
                    float,
                    int,
                ]:
                    meta["equipment"][item] = np.nan

                else:
                    if item == "moisture":
                        meta["equipment"][item] = meta["equipment"][item] / 100
                    elif item == "pore":
                        meta["equipment"][item] = meta["equipment"][item] / 3000

                item_list.append(meta["equipment"][item])

        return item_list
