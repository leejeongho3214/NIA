import copy
import errno
import random
import unicodedata
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
from torch.utils.data import ConcatDataset, Dataset


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

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
    ]
)


# 유니코드 정규화 함수
def normalize_unicode(s):
    return unicodedata.normalize("NFC", s)


class IEC_dataset(Dataset):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.area_img_list = defaultdict(list)
        self.load_img()
        self.area_img_list = dict(self.area_img_list)
        self.dig_list = None

    def __len__(self):
        return len(self.dig_list)

    def __getitem__(self, idx):
        return self.dig_list[idx]

    def load_dataset(self, dig):
        self.dig_list = self.area_img_list[dig]
        return self.dig_list

    def load_img(self):
        # with open("IEC_korea_data/test.json", "r", encoding="utf-8") as f:
        #     bbox_anno = json.load(f)
        
        with open("IEC_korea_data/test4.json", "r", encoding="utf-8") as f:
            bbox_anno = json.load(f)

        img_path_list = list()
        # for root, _, files in os.walk("IEC_korea_data"):
        #     for file in files:
        #         if ".jpg" in file or ".JPG" in file:
        #             img_path_list.append(os.path.join(root, file))
        
        for root, _, files in os.walk("IEC_korea_data/test"):
            for file in files:
                if ".jpg" in file or ".JPG" in file:
                    img_path_list.append(os.path.join(root, file))

        # img_path랑 anno key 값에 한글이 있어서 정규화 시켜줘서 에러 방지
        normalized_bbox_anno = {normalize_unicode(k): v for k, v in bbox_anno.items()}

        for img_path in tqdm(img_path_list):
            pil_img = cv2.imread(img_path)
            query = "/".join(img_path.split("/")[1:])
            normalized_query = normalize_unicode(query)
            if normalized_query in normalized_bbox_anno:
                for area, bbox in normalized_bbox_anno[normalized_query].items():
                    # 이미지 크기가 400 x 800에서 bbox 좌표를 받은거라 튜닝해줌
                    # 현재 이미지는 3000 x 5000 이미지
                    bbox[0], bbox[2] = bbox[0] * 8.64, bbox[2] * 8.64
                    bbox[1], bbox[3] = bbox[1] * 6.48, bbox[3] * 6.48
                    bbox = list(map(int, bbox))
                    bbox = [0 if a < 0 else a for a in bbox]
                    crop_img = pil_img[bbox[1] : bbox[3], bbox[0] : bbox[2]]

                    pil = Image.fromarray(crop_img.astype(np.uint8))
                    patch_img = transform_test(pil)
                    self.area_img_list[area].append([patch_img, img_path])

                pil = Image.fromarray(pil_img.astype(np.uint8))
                patch_img = transform_test(pil)
                self.area_img_list["face"].append([patch_img, img_path])


class CustomDataset_class(Dataset):
    def __init__(self, args, logger, mode):
        self.args = args
        self.logger = logger
        self.load_list(args, mode)
        self.generate_datasets()

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return idx_list[idx], self.sub_path[idx_list[idx]], self.train_num

    def generate_datasets(self):
        # train_list, val_list, test_list = list(), list(), list()
        train_list, val_list, test_list = (
            defaultdict(lambda: defaultdict()),
            defaultdict(lambda: defaultdict()),
            defaultdict(lambda: defaultdict()),
        )
        for dig, class_dict in self.json_dict.items():
            for grade, grade_dict in class_dict.items():
                random_list = list(grade_dict.keys())
                random.shuffle(random_list)

                t_len, v_len = int(len(grade_dict) * 0.8), int(len(grade_dict) * 0.1)
                t_idx, v_idx, test_idx = (
                    random_list[:t_len],
                    random_list[t_len : t_len + v_len],
                    random_list[t_len + v_len :],
                )
                grade_dict = dict(grade_dict)

                for idx_list, out_list in zip(
                    [t_idx, v_idx, test_idx], [train_list, val_list, test_list]
                ):
                    t_list = [grade_dict[idx] for idx in idx_list]
                    out_list[dig][grade] = t_list



                

        self.train_list, self.val_list, self.test_list = train_list, val_list, test_list
        # self.train_list, self.val_list, self.test_list = (
        #     ConcatDataset(train_list),
        #     ConcatDataset(val_list),
        #     ConcatDataset(test_list),
        # )

    def load_list(self, args, mode="train"):
        self.mode = mode
        target_list = [
            "pigmentation",
            "moisture",
            "elasticity_R2",
            "wrinkle_Ra",
            "pore",
        ]
        self.img_path = args.img_path
        self.json_path = args.json_path

        sub_path_list = [
            item
            for item in natsort.natsorted(os.listdir(self.img_path))
            if not item.startswith(".")
        ]

        self.json_dict = (
            defaultdict(lambda: defaultdict(list))
            if self.args.mode == "regression"
            else defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        self.json_dict_train = copy.deepcopy(self.json_dict)

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

                ## Classifying meaningful images for training from various angles of images
                for j_name in os.listdir(folder_path):
                    if self.should_skip_image(j_name, equ_name):
                        continue

                    # Load the json file for image
                    with open(os.path.join(folder_path, j_name), "r") as f:
                        json_meta = json.load(f)
                        self.process_json_meta(
                            json_meta, j_name, pre_name, equ_name, target_list, sub_fold
                        )

    def process_json_meta(
        self, json_meta, j_name, pre_name, equ_name, target_list, sub_fold
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
                dig, area = dig_n.split("_")[-1], dig_n.split("_")[-2]

                if dig in ["wrinkle", "pigmentation"] and self.mode == "test":
                    dig = f"{dig}_{area}"

                if equ_name == "01":
                    self.json_dict[dig][str(grade)][sub_fold].append(
                        [os.path.join(pre_name, j_name.split(".")[0]), meta_v]
                    )
                else:
                    self.json_dict_train[dig][str(grade)][sub_fold].append(
                        [os.path.join(pre_name, j_name.split(".")[0]), meta_v]
                    )
        else:
            for dig_n, value in json_meta["equipment"].items():
                matching_dig = [
                    dig_n for target_dig in target_list if target_dig in dig_n
                ]
                if matching_dig:
                    dig = matching_dig[0]
                    if equ_name == "01":
                        self.json_dict[dig][sub_fold].append(
                            [
                                os.path.join(pre_name, j_name.split(".")[0]),
                                value,
                                meta_v,
                            ]
                        )
                    else:
                        self.json_dict_train[dig][sub_fold].append(
                            [
                                os.path.join(pre_name, j_name.split(".")[0]),
                                value,
                                meta_v,
                            ]
                        )

    def save_dict(self, transform, flip=False, num=-1):
        pil_img = cv2.imread(os.path.join("dataset/cropped_img", self.i_path + ".jpg"))

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
            norm_v = self.norm_reg(self.value)
            label_data = norm_v

        def func():
            pil = Image.fromarray(pil_img.astype(np.uint8))
            patch_img = transform(pil)

            if patch_img.dim() == 4:
                [
                    self.area_list.append(
                        [patch_img[j], label_data, desc_area, self.dig, self.meta_v]
                    )
                    for j in range(len(patch_img[:num]))
                ]
            else:
                self.area_list.append(
                    [patch_img, label_data, desc_area, self.dig, self.meta_v]
                )

        if flip:
            for j in range(2):
                if j:
                    pil_img = cv2.flip(pil_img, flipCode=1)
                func()
        else:
            func()

    def should_skip_image(self, j_name, equ_name):
        if equ_name == "01":
            return j_name.split("_")[2] != "F"

        elif equ_name == "02":
            return (
                (
                    j_name.split("_")[2] == "F"
                    and j_name.split("_")[3].split(".")[0] in ["03", "04"]
                )
                or (
                    j_name.split("_")[2] == "L"
                    and j_name.split("_")[3].split(".")[0]
                    in ["00", "01", "03", "05", "07", "08"]
                )
                or (
                    j_name.split("_")[2] == "R"
                    and j_name.split("_")[3].split(".")[0]
                    in ["00", "01", "04", "06", "07", "08"]
                )
            )
        return False
    
    # def add_dataset(self):
    #     for a, b, c in os.walk("dataset/diff_img"):
    #         for os.path.join()
            

                # if dig == "dryness" and grade in ["0", "4"]:
                #     for a, b, c in os.walk(f"dataset/diff_img/{grade}"):
    def load_dataset(self, mode, dig_k):
        data_list = (
            self.train_list
            if mode == "train"
            else self.val_list if mode == "val" else self.test_list
        )
        self.area_list = list()
        self.dig = dig_k

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.args.res, antialias=True),
            ]
        )

        def func_v():
            transform_list = [transform_test]

            if mode == "train":
                for transform in transform_list:
                    self.save_dict(transform, True)
            else:
                self.save_dict(transform_test)

        
        if self.args.mode == "class":
            data_list = dict(data_list)
            for self.grade, class_dict in tqdm(
                data_list[dig_k].items(), desc=f"{mode}_class"
            ):
                for self.idx, sub_folder in enumerate(
                    tqdm(sorted(class_dict), desc=f"{self.dig}_{self.grade}")
                ):
                    if self.idx == self.args.data_num:
                        break

                    for self.i_path, self.meta_v in sub_folder:
                        func_v()

        else:
            for self.dig, v_list in tqdm(data_list.datasets, desc=f"{mode}_regression"):
                # if self.dig in dig_k:
                if dig_k in self.dig:
                    for vv_list in tqdm(v_list, desc=f"{self.dig}"):
                        for idx, (self.i_path, self.value, self.meta_v) in enumerate(
                            sorted(vv_list)
                        ):
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
        self.generate_datasets()

    def generate_datasets(self):
        train_list, val_list, test_list = list(), list(), list()
        for dig, v_list in self.json_dict.items():
            random_list = list(self.json_dict[dig].keys())
            random.shuffle(random_list)

            t_len, v_len = int(len(v_list) * 0.8), int(len(v_list) * 0.1)
            t_idx, v_idx, test_idx = (
                random_list[:t_len],
                random_list[t_len : t_len + v_len],
                random_list[t_len + v_len :],
            )
            v_list = dict(v_list)

            for idx_list, out_list in zip(
                [t_idx, v_idx, test_idx], [train_list, val_list, test_list]
            ):
                t_list = [v_list[idx] for idx in idx_list]
                out_list.append([dig, t_list])
                if len(self.json_dict_train[dig]) > 0:
                    tt_list = [
                        sub_list
                        for idx in idx_list
                        for sub_list in self.json_dict_train[dig][idx]
                    ]
                    out_list.append([dig, tt_list])

        self.train_list, self.val_list, self.test_list = (
            ConcatDataset(train_list),
            ConcatDataset(val_list),
            ConcatDataset(test_list),
        )


"""
pigmentation_0 -> 406
pigmentation_1 -> 1355
pigmentation_2 -> 789
pigmentation_3 -> 602
pigmentation_4 -> 198
pigmentation_5 -> 82
wrinkle_0 -> 374
wrinkle_1 -> 1566
wrinkle_2 -> 777
wrinkle_3 -> 644
wrinkle_4 -> 327
wrinkle_5 -> 321
wrinkle_6 -> 273
pore_0 -> 45
pore_1 -> 297
pore_2 -> 1035
pore_3 -> 221
pore_4 -> 92
pore_5 -> 18
sagging_0 -> 794
sagging_1 -> 346
sagging_2 -> 222
sagging_3 -> 200
sagging_4 -> 88
sagging_5 -> 56
sagging_6 -> 2
dryness_0 -> 18
dryness_1 -> 141
dryness_2 -> 520
dryness_3 -> 158
dryness_4 -> 19
ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
pigmentation_0 -> 1218
pigmentation_1 -> 1355
pigmentation_2 -> 789
pigmentation_3 -> 1204
pigmentation_4 -> 1188
pigmentation_5 -> 984
wrinkle_0 -> 1496
wrinkle_1 -> 1566
wrinkle_2 -> 1554
wrinkle_3 -> 1288
wrinkle_4 -> 1308
wrinkle_5 -> 1284
wrinkle_6 -> 1365
pore_0 -> 540
pore_1 -> 891
pore_2 -> 1035
pore_3 -> 884
pore_4 -> 1104
pore_5 -> 216
sagging_0 -> 794
sagging_1 -> 692
sagging_2 -> 666
sagging_3 -> 800
sagging_4 -> 880
sagging_5 -> 672
sagging_6 -> 24
dryness_0 -> 216
dryness_1 -> 423
dryness_2 -> 520
dryness_3 -> 474
dryness_4 -> 228
"""
