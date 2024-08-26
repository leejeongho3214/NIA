# %%
import argparse
from pytorch_grad_cam import (
    GradCAM,
)
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

git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument(
        "--img_path",
        default="dataset/img",
        type=str,
    )

    parser.add_argument(
        "--load_name",
        default=None,
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--stop_early", type=int, default=30)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--json_path",
        default="dataset/label",
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default=f"checkpoint/{git_name}",
        type=str,
    )

    parser.add_argument(
        "--aug",
        default=None,
        nargs="+",
        choices=["jitter", "crop"],
        type=str,
    )

    parser.add_argument(
        "--epoch",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--smooth",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--data_num",
        default=-1,
        type=int,
    )

    parser.add_argument(
        "--load_epoch",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )
    
    parser.add_argument(
        "--dropout",
        default = 0.3,
        type = float,
    )

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--meta", action="store_true")
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--transfer", action="store_true")

    args = parser.parse_args()

    return args

args = parse_args()

fix_seed(523)
def resume_checkpoint(model, path):
    state_dict = torch.load(path, map_location="cuda")
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict

    return model

# %%
model_num_class = (
        {"dryness": 5, "pigmentation": 6, "pore": 6, "sagging": 7, "wrinkle": 7}
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
    key: models.resnet50(args = None)
    for key, _ in model_num_class.items()
}

for key, model in model_list.items(): 
    model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
    model_list.update({key: model})
    
git_name = os.popen("git branch --show-current").readlines()[0].rstrip()
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

model_area_dict = ({
    "dryness": ["dryness"],
    "pigmentation": ["pigmentation_forehead", "pigmentation_cheek"],
    "pore": ["pore"],
    "sagging": ["sagging"],
    "wrinkle": ["wrinkle_forehead", "wrinkle_glabellus", "wrinkle_perocular"],
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


for key in model_list:
    model = model_list[key].cuda()
    model.eval()
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])

    for w_key in model_area_dict[key]:
        testset_loader, _ = dataset.load_dataset("test", w_key)
        loader_datalist = data.DataLoader(
            dataset=testset_loader,
            batch_size=32,
            num_workers=8,
            shuffle=False,
        )
        for idx, (img, label, img_name, _, _, pil_imgs) in enumerate(
            tqdm(loader_datalist, desc=w_key)
        ):
            grayscale_cams = cam(input_tensor=img, targets=None)
            img = img.detach().cpu()
            for j in range(len(grayscale_cams) // 8):
                v_img = list()
                if len(grayscale_cams) % 8 != 0 and j == len(grayscale_cams) // 8 - 1:
                    k = len(grayscale_cams) % 8
                else:
                    k = 8
                for i in range(k):
                    i = j * 8 + i
                    grayscale_cam = grayscale_cams[i, :]
                    pil_img = np.array(pil_imgs[i, :] / pil_imgs[i, :].max())
                    v_img.append(
                        show_cam_on_image(
                            pil_img, grayscale_cam, use_rgb=False
                        )
                    )
                    v_img.append(pil_img * 255.0)
                    
                    
                stacked_images = np.stack(v_img[::-1], axis=0)
                c_img = (
                    make_grid(torch.tensor(stacked_images).permute(0, 3, 1, 2), nrow=4)
                    .permute(1, 2, 0)
                    .numpy()
                )
                
                path = f"cam_output/GradCAM/{args.mode}/{args.name}"
                if not os.path.isdir(f"{path}/{w_key}"):
                    mkdir(f"{path}/{w_key}")
                    
                cv2.imwrite(f"{path}/{w_key}/{idx}_{j}.jpg",c_img)
