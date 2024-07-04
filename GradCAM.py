# %%
from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    EigenCAM,
    FullGrad,
)
import GPUtil
import torch.nn as nn
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tool.utils import fix_seed, mkdir
from torchvision import models
import os
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import random_split, ConcatDataset, Dataset
import natsort
from collections import defaultdict
from tqdm import tqdm
import json
from torch.utils import data


fix_seed(523)
# %%
def resume_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cuda")
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict

    return model

# %%
model_num_class = {
    "dryness": 5,
    "pigmentation": 6,
    "pore": 6,
    "sagging": 7,
    "wrinkle": 7,
}


model_list = {
    key: models.resnet50(args = None)
    for key, _ in model_num_class.items()
}

for key, model in model_list.items(): 
    model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
    model_list.update({key: model})
    
    
name = "cnn_cb_aug_1"
task = "class"

if len(os.popen("git branch --show-current").readlines()):
    git_name = os.popen("git branch --show-current").readlines()[0].rstrip()
else:
    git_name = os.popen("git describe --tags").readlines()[0].rstrip()

## Adjust the number of output in model for each region image
check_path = os.path.join("checkpoint", git_name, task, name, "save_model")

if os.path.isdir(check_path):
    for key, model in model_list.items():
        model_list[key] = resume_checkpoint(
            model,
            os.path.join(check_path, f"{key}", "state_dict.bin"),
        )
        print(f"success => {key}")


# %%
import copy
import random
import torch.nn.functional as F


class CustomDataset_class(Dataset):
    def __init__(self, mode):
        self.load_list(mode)
        self.generate_datasets()

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        idx_list = list(self.sub_path.keys())
        return self.sub_path[idx_list[idx]]

    def generate_datasets(self):
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

    def load_list(self, mode="train"):
        self.mode = mode
        target_list = [
            "pigmentation",
            "moisture",
            "elasticity_R2",
            "wrinkle_Ra",
            "pore",
        ]
        self.img_path = "dataset/img"
        self.json_path = "dataset/label"

        sub_path_list = [
            item
            for item in natsort.natsorted(os.listdir(self.img_path))
            if not item.startswith(".")
        ]

        self.json_dict = (
            defaultdict(lambda: defaultdict(list))
            if task == "regression"
            else defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        self.json_dict_train = copy.deepcopy(self.json_dict)

        for equ_name in sub_path_list:
            if equ_name.startswith(".") or int(equ_name) != 1:
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
                    if self.should_skip_image(j_name, equ_name):
                        continue

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
            and task == "class"
        ) or (
            (json_meta["equipment"] is None or json_meta["images"]["bbox"] is None)
            and task == "regression"
        ):
            return

        age = torch.tensor([round((json_meta["info"]["age"] - 13.0) / 69.0, 5)])
        gender = 0 if json_meta["info"]["gender"] == "F" else 1
        gender_l = F.one_hot(torch.tensor(gender), 2)
        meta_v = torch.cat([age, gender_l])

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

    def save_dict(self, transform, flip=False, num=-1):
        pil_img = cv2.imread(os.path.join("dataset/cropped_img", self.i_path + ".jpg"))
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)

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

        if task == "class":
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
                        [patch_img[j], label_data, desc_area, self.dig, pil_img]
                    )
                    for j in range(len(patch_img[:num]))
                ]
            else:
                self.area_list.append(
                    [patch_img, label_data, desc_area, self.dig, pil_img]
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
            return (
                (
                    j_name.split("_")[2] == "Ft"
                    and j_name.split("_")[3].split(".")[0]
                    in ["00", "01", "02", "03", "04", "05", "06", "07"]
                )
                or (
                    j_name.split("_")[2] == "Fb"
                    and j_name.split("_")[3].split(".")[0]
                    in ["00", "02", "03", "04", "05", "06", "07", "08"]
                )
                or (
                    j_name.split("_")[2] == "F"
                    and j_name.split("_")[3].split(".")[0] in ["03", "04", "05", "06"]
                )
                or (j_name.split("_")[2] in ["R15", "L15"])
                or (
                    j_name.split("_")[2] == "R30"
                    and j_name.split("_")[3].split(".")[0]
                    in ["00", "01", "02", "04", "06", "07", "08"]
                )
                or (
                    j_name.split("_")[2] == "L30"
                    and j_name.split("_")[3].split(".")[0]
                    in ["00", "01", "02", "03", "05", "07", "08"]
                )
            )
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
                transforms.Resize((256, 256), antialias=True),
            ]
        )

        def func_v():
            self.save_dict(transform_test)

        data_list = dict(data_list)
        for self.grade, class_dict in tqdm(
            data_list[dig_k].items(), desc=f"{mode}_class"
        ):
            for self.idx, sub_folder in enumerate(
                tqdm(sorted(class_dict), desc=f"{self.dig}_{self.grade}")
            ):
                for self.i_path, self.meta_v in sub_folder:
                    func_v()

        return self.area_list


# %%
# %%
dataset = CustomDataset_class("test")
model_area_dict = {
    "dryness": ["dryness"],
    "pigmentation": ["pigmentation_forehead", "pigmentation_cheek"],
    "pore": ["pore"],
    "sagging": ["sagging"],
    "wrinkle": ["wrinkle_forehead", "wrinkle_glabellus", "wrinkle_perocular"],
}
import gc
from torchvision.utils import make_grid
from GPUtil import showUtilization as gpu_usage


for key in model_list:
    model = model_list[key].cuda()
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    for w_key in model_area_dict[key]:
        testset_loader = dataset.load_dataset("test", w_key)
        loader_datalist = data.DataLoader(
            dataset=testset_loader,
            batch_size=32 if model == "cnn" else 8,
            num_workers=8,
            shuffle=False,
        )
        for idx, (img, label, img_name, _, pil_imgs) in enumerate(
            tqdm(loader_datalist, desc=w_key)
        ):
            v_img = list()
            grayscale_cams = cam(input_tensor=img, targets=None)
            for i in range(len(grayscale_cams)):
                grayscale_cam = grayscale_cams[i, :]
                pil_img = np.array(pil_imgs[i, :] / pil_imgs[i, :].max())
                v_img.append(
                    show_cam_on_image(
                        pil_img[:, :, (2, 1, 0)], grayscale_cam, use_rgb=True
                    )
                )
                
            im = np.clip(np.array(img.permute(0, 2, 3, 1)) * 255, 0, 255).astype(np.int32)
            for i in range(len(im)):
                v_img.append(im[i])
                
            stacked_images = np.stack(v_img, axis=0)
            c_img = (
                make_grid(torch.tensor(stacked_images).permute(0, 3, 1, 2), nrow=4)
                .permute(1, 2, 0)
                .numpy()
            )
            
            path = f"cam_output/GradCAM/{name}"
            if not os.path.isdir(f"{path}/{w_key}"):
                mkdir(f"{path}/{w_key}")
                
            plt.figure(dpi=600)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(c_img)
            plt.savefig(f"{path}/{w_key}/{idx}.jpg")
            
            torch.cuda.empty_cache()
            gc.collect()
            if idx == 3:
                break

    
