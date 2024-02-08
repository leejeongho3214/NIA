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
from torch.utils.data import random_split, ConcatDataset, Dataset

from utils import noramlize_v

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

class CustomDataset_class(Dataset):
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
        target_list = [
            "pigmentation",
            "forehead_moisture",
            "forehead_elasticity_R2",
            "perocular_wrinkle_Ra",
            "cheek_moisture",
            "cheek_elasticity_R2",
            "cheek_pore",
            "chin_moisture",
            "chin_elasticity_R2",
        ]
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
                natsort.natsorted(os.listdir(os.path.join("dataset/label", equ_name))),
                desc="path loading..",
            ):
                pre_name = os.path.join(equ_name, sub_fold)
                folder_path = os.path.join("dataset/label", pre_name)

                if sub_fold.startswith(".") or not os.path.exists(
                    os.path.join(self.json_path, pre_name)
                ):
                    continue
                
                
                for j_name in os.listdir(folder_path):
                    if j_name.split("_")[3].split(".")[0] != "00": continue
                    with open(os.path.join(folder_path, j_name), "r") as f:
                        json_meta = json.load(f)

                        img = cv2.imread(
                            f"dataset/img/{equ_name}/{json_meta['info']['id']}/{json_meta['info']['filename']}"
                        )
                        # bbox = json_meta["images"]["bbox"]
                        # if bbox == None:
                        #     continue
                        
                        # max_l = max(bbox[3] - bbox[1], bbox[2] - bbox[0]) // 2
                        # center_p = ((bbox[1] + bbox[3]) // 2, (bbox[0] + bbox[2]) // 2)
                        # p_img = img[
                        #     max(0, center_p[0] - max_l) : min(
                        #         img.shape[0] - 1, center_p[0] + max_l
                        #     ),
                        #     max(0, center_p[1] - max_l) : min(
                        #         img.shape[1] - 1, center_p[1] + max_l
                        #     ),
                        # ]
                        # r_value = 256 / max(p_img.shape)

                        # pil_img = np.zeros([256, 256, 3])
                        # r_img = cv2.resize(
                        #     p_img,
                        #     (
                        #         int(p_img.shape[1] * r_value),
                        #         int(p_img.shape[0] * r_value),
                        #     ),
                        # )

                        # pil_img[: r_img.shape[0], : r_img.shape[1]] = r_img
                        
                        pil_img = cv2.resize(img, (256, 256))
                        
                        
                        mkdir(f"dataset/cropped_img/{equ_name}/{json_meta['info']['id']}")
                        cv2.imwrite(
                            f"dataset/cropped_img/{equ_name}/{json_meta['info']['id']}/{json_meta['info']['filename'].split('.')[0]}_{json_meta['images']['facepart']:02}.jpg",
                            pil_img,
                        )
                        
                        continue
         
        assert 0, "finish"           
                # ## Classifying meaningful images for training from various angles of images
                # for j_name in os.listdir(folder_path):
                #     if equ_name == "01":
                #         if (
                #             (
                #                 j_name.split("_")[2] == "Ft"
                #                 and j_name.split("_")[3].split(".")[0]
                #                 in ["01", "02", "03", "04"]
                #             )
                #             or (
                #                 j_name.split("_")[2] == "Fb"
                #                 and j_name.split("_")[3].split(".")[0]
                #                 in ["01", "02", "03", "04", "05", "06", "07", "08"]
                #             )
                #             or (
                #                 j_name.split("_")[2] == "F"
                #                 and j_name.split("_")[3].split(".")[0] in ["03", "04"]
                #             )
                #         ):
                #             continue

                #         elif (
                #             j_name.split("_")[2].startswith("R")
                #             and j_name.split("_")[3].split(".")[0] in ["04", "06"]
                #         ) or (
                #             j_name.split("_")[2].startswith("L")
                #             and j_name.split("_")[3].split(".")[0] in ["03", "05"]
                #         ):
                #             continue

                #         elif j_name.split("_")[2] in [
                #             "R30",
                #             "L30",
                #         ] and j_name.split(
                #             "_"
                #         )[3].split(".")[0] in ["01", "02"]:
                #             continue

                #     elif equ_name == "02":
                #         if (
                #             (
                #                 j_name.split("_")[2] == "F"
                #                 and j_name.split("_")[3].split(".")[0]
                #                 not in ["00", "01", "02", "05", "06", "07", "08"]
                #             )
                #             or (
                #                 j_name.split("_")[2] == "L"
                #                 and j_name.split("_")[3].split(".")[0]
                #                 not in ["02", "04", "06"]
                #             )
                #             or (
                #                 j_name.split("_")[2] == "R"
                #                 and j_name.split("_")[3].split(".")[0]
                #                 not in ["02", "03", "05"]
                #             )
                #         ):
                #             continue

                #     with open(os.path.join(folder_path, j_name), "r") as f:
                #         json_meta = json.load(f)

                #         if (
                #             (
                #                 list(json_meta["annotations"].keys())[0] == "acne"
                #                 or json_meta["images"]["bbox"] == None
                #             )
                #             and args.mode == "class"
                #         ) or (
                #             (
                #                 json_meta["equipment"] == None
                #                 or json_meta["images"]["bbox"] == None
                #             )
                #             and args.mode == "regression"
                #         ):
                #             continue
                        
                #         if json_meta["images"]["bbox"] == None: 
                #             continue

                #         age = torch.tensor(
                #             [round((json_meta["info"]["age"] - 13.0) / 69.0, 5)]
                #         )
                #         gender = 0 if json_meta["info"]["gender"] == "F" else 1
                #         gender_l = F.one_hot(torch.tensor(gender), 2)

                #         meta_v = torch.concat([age, gender_l])

                #         if args.mode == "class":
                #             for dig_n, grade in json_meta["annotations"].items():
                #                 dig, area = (
                #                     dig_n.split("_")[-1],
                #                     dig_n.split("_")[-2],
                #                 )

                #                 if dig in [
                #                     "wrinkle",
                #                     "pigmentation",
                #                 ]:
                #                     dig = f"{dig}_{area}"

                #                 if equ_name == "01":
                #                     self.json_dict[dig][str(grade)].append(
                #                         [
                #                             os.path.join(
                #                                 pre_name, j_name.split(".")[0]
                #                             ),
                #                             meta_v,
                #                         ]
                #                     )
                #                 else:
                #                     self.json_dict_train[dig][str(grade)].append(
                #                         [
                #                             os.path.join(
                #                                 pre_name, j_name.split(".")[0]
                #                             ),
                #                             meta_v,
                #                         ]
                #                     )
                #         else:
                #             for dig_n, value in json_meta["equipment"].items():
                #                 matching_dig = [
                #                     target_dig
                #                     for target_dig in target_list
                #                     if target_dig in dig_n
                #                 ]
                #                 if len(matching_dig) != 0:
                #                     dig = matching_dig[0]

                #                     if equ_name == "01":
                #                         self.json_dict[dig][sub_fold].append(
                #                             [
                #                                 os.path.join(
                #                                     pre_name, j_name.split(".")[0]
                #                                 ),
                #                                 value,
                #                                 meta_v,
                #                             ]
                #                         )
                #                     else:
                #                         self.json_dict_train[dig][sub_fold].append(
                #                             [
                #                                 os.path.join(
                #                                     pre_name, j_name.split(".")[0]
                #                                 ),
                #                                 value,
                #                                 meta_v,
                #                             ]
                #                         )

    def save_dict(self, transform):
        p_img = cv2.imread(os.path.join("dataset/cropped_img", self.i_path + ".jpg"))
        r_value = 256 / max(p_img.shape)

        pil_img = np.zeros([256, 256, 3])
        r_img = cv2.resize(
            p_img,
            (int(p_img.shape[1] * r_value), int(p_img.shape[0] * r_value)),
        )
        pil_img[: r_img.shape[0], : r_img.shape[1]] = r_img

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
        pil = Image.fromarray(pil_img.astype(np.uint8))
        patch_img = transform(pil)
        if self.args.mode == "class":
            label_data = int(self.grade)
        else:
            norm_v = self.norm_reg(self.value)
            label_data = norm_v

        if patch_img.dim() == 4:
            [
                self.area_list.append(
                    [patch_img[j], label_data, desc_area, self.dig, self.meta_v]
                )
                for j in range(len(patch_img))
            ]
        else:
            self.area_list.append(
                [patch_img, label_data, desc_area, self.dig, self.meta_v]
            )

        return pil_img

    def load_dataset(self, mode, dig_k):
        data_list = (
            self.train_list
            if mode == "train"
            else self.val_list
            if mode == "val"
            else self.test_list
        )
        self.area_list = list()

        transform_test = transforms.Compose(
            [transforms.ToTensor(), noramlize_v(dig_k, self)]
        )

        transform_crop = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.FiveCrop(230),
                transforms.Lambda(
                    lambda crops: torch.stack([(crop) for crop in crops])
                ),
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
                transforms.Resize(256, antialias=True),
                noramlize_v(dig_k, self),
            ]
        )
        aug_naming = {"crop": transform_crop, "jitter": transform_color}

        if self.args.aug == None:
            transform_list = [transform_test]
        else:
            transform_list = [transform_test] + [
                aug_naming[t_name] for t_name in self.args.aug
            ]

        self.logger.debug(transform_list)

        def func_v():
            if mode == "train":
                for transform in transform_list:
                    pil_img = self.save_dict(transform)
                    pil_img = cv2.flip(pil_img, flipCode=1)
                    self.save_dict(transform)
            else:
                self.save_dict(transform_test)

        if self.args.mode == "class":
            for self.dig, self.grade, class_dict in tqdm(
                data_list.datasets, desc=f"{mode}_class"
            ):
                if self.dig == dig_k:
                    for idx, (self.i_path, self.meta_v) in enumerate(
                        tqdm(sorted(class_dict), desc=f"{self.dig}_{self.grade}")
                    ):
                        if idx == self.args.data_num:
                            break
                        func_v()

        else:
            for self.dig, v_list in tqdm(data_list.datasets, desc=f"{mode}_regression"):
                if self.dig == dig_k:
                    for idx, (self.i_path, self.value, self.meta_v) in enumerate(
                        tqdm(sorted(v_list), desc=f"{self.dig}")
                    ):
                        if idx == self.args.data_num:
                            break
                        func_v()

        return self.area_list

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
        self.load_list(args)
        train_list, val_list, test_list = list(), list(), list()
        self.json_dict_train = dict(self.json_dict_train)

        for dig, v_list in self.json_dict.items():
            random_list = list(self.json_dict[dig].keys())
            random.shuffle(random_list)
            
            t_len, v_len = int(len(v_list) * 0.8), int(len(v_list) * 0.1)
            t_idx = random_list[: t_len]
            v_idx = random_list[t_len: t_len + v_len]
            test_idx = random_list[t_len + v_len:]
            
            v_list = dict(v_list)
            
            t_list = [sub_list for idx in t_idx for sub_list in v_list[idx]]
            train_list.append([dig, t_list])
            v_list2 = [sub_list for idx in v_idx for sub_list in v_list[idx]]
            val_list.append([dig, v_list2])
            t_list2 = [sub_list for idx in test_idx for sub_list in v_list[idx]]
            test_list.append([dig, t_list2])
            
            if len(self.json_dict_train[dig]) > 0:
                tt_list = [sub_list for idx in t_idx for sub_list in self.json_dict_train[dig][idx]]
                train_list.append([dig, tt_list])

        self.train_list, self.val_list, self.test_list = (
            ConcatDataset(train_list),
            ConcatDataset(val_list),
            ConcatDataset(test_list),
        )
