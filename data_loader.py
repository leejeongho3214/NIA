from PIL import Image
import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
from tqdm import tqdm

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
    "0": "all",
    "1": "forearm",
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

regression_name = {}


class CustomDataset(Dataset):
    def __init__(self, args):
        self.img_path = args.img_path
        self.json_path = args.json_path
        sub_path_list = os.listdir(self.img_path)
        sub_path_list = [f"{equ:02}" for equ in args.equ]
        element = [
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = {
            "train": transforms.Compose(element),
            "test": transforms.Compose(element),
        }
        self.sub_path = list()
        for equ_name in tqdm(sub_path_list, desc="equ_name"):
            for sub_fold in tqdm(
                os.listdir(os.path.join(self.img_path, equ_name)),
                desc="subject_no",
                leave=False,
            ):
                if sub_fold.startswith(".") or not os.path.exists(
                    os.path.join(self.json_path, equ_name, sub_fold)
                ):
                    continue

                img_count = 0
                for file_name in os.listdir(
                    os.path.join(self.img_path, equ_name, sub_fold)
                ):
                    if any(
                        file_name.lower().endswith(ext) for ext in [".jpg", ".jpeg"]
                    ):
                        img_count += 1
                if img_count == img_num[equ_name]:
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
                        start_idx = 1 if args.mode == "class" else 0
                        for idx_area in range(start_idx, 9):
                            try:
                                (
                                    reduction_value,
                                    json_name,
                                    area_name,
                                    meta,
                                    ori_patch_img,
                                ) = self.load_img(
                                    img_name, angle, idx_area, equ_name, img, args
                                )
                            except:
                                if self.load_img(
                                    img_name, angle, idx_area, equ_name, img, args
                                ):
                                    area_list[f"{idx_area}"] = [
                                        torch.zeros([3, 128, 128]),
                                        dict(),
                                    ]
                                    continue
                            if idx_area != 0:
                                try:
                                    n_patch_img = cv2.resize(
                                        ori_patch_img,
                                        (
                                            int(
                                                ori_patch_img.shape[1] / reduction_value
                                            ),
                                            int(
                                                ori_patch_img.shape[0] / reduction_value
                                            ),
                                        ),
                                    )
                                except:
                                    print(
                                        f"Sub No: {json_name.split('_')[0]} & Angle: {angle} & Equ: {equ_name} & Area: {area_naming[area_name]} , {list(ori_patch_img.shape)}"
                                    )
                                    area_list[f"{idx_area}"] = [
                                        torch.zeros([3, 128, 128]),
                                        dict(),
                                    ]
                                    continue

                                patch_img = self.make_double(n_patch_img)
                                if not isinstance(patch_img, np.ndarray):
                                    continue

                            else:
                                patch_img = cv2.resize(
                                    ori_patch_img, (args.res, args.res)
                                )

                            pil_img = Image.fromarray(patch_img)
                            patch_img = self.transform["train"](pil_img)
                            label_data = (
                                meta["annotations"]
                                if args.mode == "class"
                                else meta["equipment"]
                            )
                            if type(label_data) != dict:
                                continue

                            if args.mode != "class":
                                label_data = torch.tensor(self.norm_reg(meta, idx_area))

                            area_list[f"{idx_area}"] = [
                                patch_img,
                                label_data
                            ]

                        self.sub_path.append(area_list)


                else:
                    print(
                        f"\033[94m{os.path.join(self.img_path, equ_name, sub_fold)}는 제외됨\033[0m"
                    )

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        return self.sub_path[idx]

    def make_double(self, n_patch_img):
        row = n_patch_img.shape[0]
        col = n_patch_img.shape[1]

        if row < 64:
            patch_img = np.zeros((128, 256, 3), dtype=np.uint8)
            patch = cv2.resize(
                n_patch_img,
                (
                    int(n_patch_img.shape[1] * 2),
                    int(n_patch_img.shape[0] * 2),
                ),
            )
            patch_img[
                : n_patch_img.shape[0] * 2, : int(n_patch_img.shape[1]) * 2
            ] = patch

        elif col < 64:
            patch_img = np.zeros((256, 128, 3), dtype=np.uint8)
            patch = cv2.resize(
                n_patch_img,
                (
                    int(n_patch_img.shape[1] * 2),
                    int(n_patch_img.shape[0] * 2),
                ),
            )
            patch_img[
                : n_patch_img.shape[0] * 2, : int(n_patch_img.shape[1] * 2)
            ] = patch

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
                json_name.split("_")[0],
                json_name,
            ),
            "r",
            encoding="utf8",
        ) as f:
            meta = json.load(f)

        if any([item == "None" for item in meta["images"]["bbox"]]):
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

        if (bbox_x[1] - bbox_x[0]) < 128 or (bbox_y[1] - bbox_y[0]) < 128:
            return 1

        area_name = str(int(json_name.split("_")[-1].split(".")[0]))
        if (bbox_point[0] > bbox_point[2]) or (bbox_point[1] > bbox_point[3]):
            print(
                f"Sub No: {json_name.split('_')[0]} & Angle: {angle} & Area: {area_naming[area_name]}// 위치 반전"
            )

        patch_img = img[bbox_y[0] : bbox_y[1], bbox_x[0] : bbox_x[1]]

        reduction_value = max(patch_img.shape) / args.res


        return reduction_value, json_name, area_name, meta, patch_img

    def norm_reg(self, meta, idx_area):
        item_list = list()
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
            item_class = item.split("_")[-1]
            if meta["equipment"][item] == 'Er':
                item_list.append(np.nan)
            else:
                if item_class == "R2":
                    item_list.append(meta["equipment"][item])

                elif item_class in ["moisture", "Ra"]:
                    item_list.append(meta["equipment"][item] / 100)

                elif item_class == "count":
                    item_list.append(meta["equipment"][item] / 300)

                elif item_class == "pore":
                    item_list.append(meta["equipment"][item] / 3000)

                else:
                    assert 0, "item_class is not here"

        return item_list
