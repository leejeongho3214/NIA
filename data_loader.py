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
    "3": "l_perocular",
    "4": "r_perocular",
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


def noramlize_v(key, self):
    if key == "dryness":
        a = (
            transforms.Normalize(
                [0.3971863, 0.53899115, 0.7918039],
                [0.00548187, 0.00920583, 0.0188247],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.38747337, 0.5135743, 0.7418671],
                [0.00517165, 0.00811211, 0.01600054],
            )
        )
    elif key == "pigmentation_forehead":
        a = (
            transforms.Normalize(
                [0.35794356, 0.4875437, 0.67509633],
                [0.00749865, 0.00930441, 0.01313221],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.3644775, 0.47519714, 0.64031136],
                [0.01067846, 0.01087018, 0.01358093],
            )
        )
    elif key == "pigmentation_cheek":
        a = (
            transforms.Normalize(
                [0.4236276, 0.57461, 0.81209165],
                [0.01844954, 0.02865098, 0.0412216],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.41664535, 0.5376691, 0.7362832],
                [0.01813314, 0.02635618, 0.0385823],
            )
        )
    elif key == "pore":
        a = (
            transforms.Normalize(
                [0.4212271, 0.5735757, 0.8125185],
                [0.01824168, 0.02863191, 0.04099633],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.41947985, 0.5401359, 0.738836],
                [0.01840642, 0.02611311, 0.0376912],
            )
        )
    elif key == "sagging":
        a = (
            transforms.Normalize(
                [0.2894094, 0.39513916, 0.55362684],
                [0.0075179, 0.00886958, 0.01018988],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.2864517, 0.3769295, 0.51980555],
                [0.00982202, 0.01481378, 0.02048211],
            )
        )
    elif key == "wrinkle_forehead":
        a = (
            transforms.Normalize(
                [0.35502893, 0.4822758, 0.66742766],
                [0.00764483, 0.00957104, 0.01369465],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.35892627, 0.4686825, 0.63134724],
                [0.00991173, 0.00985187, 0.01280279],
            )
        )
    elif key == "wrinkle_glabellus":
        a = (
            transforms.Normalize(
                [0.4604649, 0.6125847, 0.85397255],
                [0.00545718, 0.00695129, 0.009476],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.41069034, 0.5170356, 0.7017539],
                [0.00623439, 0.00779748, 0.01017918],
            )
        )
    elif key == "wrinkle_perocular":
        a = (
            transforms.Normalize(
                [0.4000641, 0.56431776, 0.7999528],
                [0.0093093, 0.01803441, 0.02781325],
            )
            if "2" not in self.args.equ
            else transforms.Normalize(
                [0.4115054, 0.53571254, 0.72633845],
                [0.01143675, 0.01620491, 0.02333776],
            )
        )
    else:
        assert 0, "key name is incorrect"

    return a


class CustomDataset(Dataset):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
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

        if len(self.json_dict_train) > 0:
            for dig, class_dict in self.json_dict_train.items():
                for grade, name_list in class_dict.items():
                    if len(name_list) < 10:
                        name_list = name_list * (10 // len(name_list) + 1)
                    train_list.append([dig, grade, name_list])

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

        self.json_dict = defaultdict(lambda: defaultdict(list))
        self.json_dict_train = defaultdict(lambda: defaultdict(list))
        
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

                    ## Classifying meaningful images for training from various angles of images
                    for j_name in os.listdir(json_name):
                        if equ_name == "01":
                            if (
                                (
                                    j_name.split("_")[2] == "Ft"
                                    and j_name.split("_")[3].split(".")[0]
                                    in ["01", "02", "03", "04"]
                                )
                                or (
                                    j_name.split("_")[2] == "Fb"
                                    and j_name.split("_")[3].split(".")[0]
                                    in ["01", "02", "03", "04", "05", "06", "07", "08"]
                                )
                                or (
                                    j_name.split("_")[2] == "F"
                                    and j_name.split("_")[3].split(".")[0]
                                    in ["03", "04"]
                                )
                            ):
                                continue

                            elif (
                                j_name.split("_")[2].startswith("R")
                                and j_name.split("_")[3].split(".")[0] in ["04", "06"]
                            ) or (
                                j_name.split("_")[2].startswith("L")
                                and j_name.split("_")[3].split(".")[0] in ["03", "05"]
                            ):
                                continue

                            elif j_name.split("_")[2] in [
                                "R30",
                                "L30",
                            ] and j_name.split("_")[3].split(".")[0] in ["01", "02"]:
                                continue

                        elif equ_name == "02":
                            if (
                                (
                                    j_name.split("_")[2] == "F"
                                    and j_name.split("_")[3].split(".")[0]
                                    not in ["01", "02", "05", "06", "07", "08"]
                                )
                                or (
                                    j_name.split("_")[2] == "L"
                                    and j_name.split("_")[3].split(".")[0]
                                    not in ["02", "04", "06"]
                                )
                                or (
                                    j_name.split("_")[2] == "R"
                                    and j_name.split("_")[3].split(".")[0]
                                    not in ["02", "03", "05"]
                                )
                            ):
                                continue

                        with open(os.path.join(json_name, j_name), "r") as f:
                            json_meta = json.load(f)

                            ## Sorting frontal images and img without bbox values
                            if (
                                list(json_meta["annotations"].keys())[0] == "acne"
                                or json_meta["images"]["bbox"] == None
                            ):
                                continue
                            
                            age = torch.tensor([round((json_meta["info"]["age"] - 13.0) / 69.0, 5)])
                            gender = 0 if json_meta["info"]["gender"] == "F" else 1
                            gender_l = F.one_hot(torch.tensor(gender), 2)

                            meta_v = torch.concat([age, gender_l])
                            
                            for dig_n, grade in json_meta["annotations"].items():
                                dig, area = (
                                    dig_n.split("_")[-1],
                                    dig_n.split("_")[-2],
                                )
                                ## Segmenting regions with the same dicrease name
                                if dig in [
                                    "wrinkle",
                                    "pigmentation",
                                ]:
                                    dig = f"{dig}_{area}"

                                if equ_name == "01":
                                    self.json_dict[dig][str(grade)].append(
                                        [os.path.join(pre_name, j_name.split(".")[0]), meta_v]
                                    )
                                else:
                                    self.json_dict_train[dig][str(grade)].append(
                                        [os.path.join(pre_name, j_name.split(".")[0]), meta_v]
                                    )

    def load_dataset(self, mode, dig_k):
        sub_path = []
        data_list = (
            self.train_list
            if mode == "train"
            else self.val_list
            if mode == "val"
            else self.test_list
        )
        area_list = defaultdict(lambda: defaultdict(list))

        transform_test = transforms.Compose(
            [transforms.ToTensor(), noramlize_v(dig_k, self)]
        )


        transform_crop = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.FiveCrop(230),
                transforms.Lambda(lambda crops: torch.stack([(crop) for crop in crops])),
                transforms.Resize(256, antialias=True),
                noramlize_v(dig_k, self),
            ]
        )

        transform_color = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                transforms.ToTensor(),
                transforms.FiveCrop(230),
                transforms.Lambda(lambda crops: torch.stack([(crop) for crop in crops])),
                transforms.Resize(256, antialias=True),
                noramlize_v(dig_k, self),
            ]
        )

        transform_both = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.CenterCrop(200),
                transforms.Resize(256),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                noramlize_v(dig_k, self),
            ]
        )
        
        transform_list = [transform_test] if not self.args.jitter else [transform_test, transform_color]
        
        if dig_k == "dryness": 
            self.logger.debug(transform_list)

        def save_dict(idx, transform):
            pil = Image.fromarray(pil_img.astype(np.uint8))
            patch_img = transform(pil)
            if idx > 0: 
                [area_list[dig][grade].append([patch_img[j], int(grade), desc_area, dig, meta_v]) for j in range(len(patch_img))]
            else:
                area_list[dig][grade].append([patch_img, int(grade), desc_area, dig, meta_v])

        for dig, grade, class_dict in tqdm(data_list.datasets, desc=f"{mode}_class"):
            if dig == dig_k:
                for idx, (i_path, meta_v) in enumerate(
                    tqdm(sorted(class_dict), desc=f"{dig}_{grade}")
                ):
                    if idx == self.args.data_num:
                        break

                    p_img = cv2.imread(
                        os.path.join("dataset/cropped_img", i_path + ".jpg")
                    )
                    r_value = 256 / max(p_img.shape)

                    pil_img = np.zeros([256, 256, 3])
                    r_img = cv2.resize(
                        p_img,
                        (int(p_img.shape[1] * r_value), int(p_img.shape[0] * r_value)),
                    )
                    pil_img[: r_img.shape[0], : r_img.shape[1]] = r_img
                    # if i_path.split("_")[-1] in ["04", "06"]:
                    #     pil_img = cv2.flip(pil_img, 1)

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
                        for idx, transform in enumerate(transform_list):
                            save_dict(idx, transform)
                            pil_img = cv2.flip(pil_img, flipCode=1)
                            save_dict(idx, transform)
                    else:
                        save_dict(0, transform_test)

        sub_path = [item for items in area_list[dig_k].values() for item in items]

        del area_list

        return sub_path

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
