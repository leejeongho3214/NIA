# %%
import argparse
from pytorch_grad_cam import (
    GradCAM,
)
import sys
import os


import torch.nn as nn
from pytorch_grad_cam.utils.image import show_cam_on_image
from tool.utils import fix_seed, mkdir
from torchvision import models
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils import data
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )
    
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )
    
    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )
    
    args = parser.parse_args()

    return args

args = parse_args()

fix_seed(args.seed)

def resume_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cuda")
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict

    return model

model_num_class = (
        {
        #     "dryness": 5, 
        #  "pigmentation": 6, 
        #  "pore": 6, 
         "sagging": 6, 
         "wrinkle": 7
         }
        if args.mode == "class"
        else {
            "pigmentation": 1,
            "moisture": 1,
            "elasticity_R2": 1,
            "wrinkle_Ra": 1,
            "pore": 1,
        }
    )

model_list = {
    key: models.coatnet.coatnet_4(num_classes=value)
    for key, value in model_num_class.items()
}

for key, model in model_list.items(): 
    model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
    model_list.update({key: model})
    
check_path = os.path.join("checkpoint", git_name, args.mode, args.name, "save_model")

if os.path.isdir(check_path):
    for key, model in model_list.items():
        model_list[key] = resume_checkpoint(
            model,
            os.path.join(check_path, f"{key}", "state_dict.bin"),
        )
        print(f"success => {key}")


from tool.data_loader import CustomDataset_class, CustomDataset_regress

dataset = (
    CustomDataset_class(args, None, "test")
    if args.mode == "class"
    else CustomDataset_regress(args, None)
)
    
model_area_dict = (
    {
        "dryness": ["dryness"],
        "pigmentation": ["forehead_pigmentation", "cheek_pigmentation"],
        "pore": ["pore"],
        "sagging": ["sagging"],
        "wrinkle": ["forehead_wrinkle", "glabellus_wrinkle", "perocular_wrinkle"],
    }
    if args.mode == "class"
    else {
        "pigmentation": ["pigmentation"],
        "moisture": ["forehead_moisture", "cheek_moisture", "chin_moisture"],
        "elasticity_R2": [
            "forehead_elasticity_R2",
            "cheek_elasticity_R2",
            "chin_elasticity_R2",
        ],
        "wrinkle_Ra": ["perocular_wrinkle_Ra"],
        "pore": ["cheek_pore"],
    }
)

from torchvision.utils import make_grid
testset_loader, _ = dataset.load_dataset("test")

for key in model_list:
    model = model_list[key].cuda()
    cam = GradCAM(model=model, target_layers=[model.s2[-1]])
    
    for w_key in model_area_dict[key]:
        loader_datalist = data.DataLoader(
            dataset=testset_loader[w_key],
            batch_size=8,
            num_workers=8,
            shuffle=False,
        )
        
        for idx, (img, label, img_name, _, _, pil_imgs) in enumerate(
            tqdm(loader_datalist, desc=w_key)
        ):
            v_img = []
            grayscale_cams = cam(input_tensor=img)

            for i in range(len(grayscale_cams)):
                grayscale_cam = grayscale_cams[i, :]
                pil_img = np.array(pil_imgs[i], dtype=np.float32) / 255.0  # Normalize
                v_img.append(show_cam_on_image(pil_img, grayscale_cam, use_rgb=False))

            # 원본 이미지 추가 (deepcopy 방지)
            v_img.extend(pil_imgs)

            # numpy -> tensor 변환 (CPU에서 실행)
            stacked_images = torch.tensor(np.stack(v_img, axis=0)).permute(0, 3, 1, 2).float().cpu()

            # make_grid CPU에서 실행
            c_img = make_grid(stacked_images, nrow=4).permute(1, 2, 0).numpy()

            path = f"cam_output/GradCAM/{args.mode}/{git_name}/{args.name}"
            if not os.path.isdir(f"{path}/{w_key}"):
                mkdir(f"{path}/{w_key}")

            # 이미지 저장 최적화
            cv2.imwrite(f"{path}/{w_key}/{idx}_trans.jpg", c_img)

            if idx == 3:
                break


    
