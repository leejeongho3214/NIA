import os

import shutil
import sys
import numpy as np
import torch
from torchvision import models
from tensorboardX import SummaryWriter
from utils import mkdir, resume_checkpoint
from logger import setup_logger
from data_loader import CustomDataset
from model import Model

import argparse

git_name = os.popen('git branch --show-current').readlines()[0].rstrip()

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
        default= f"checkpoint/{git_name}",
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
        default=128,
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
        default=0.05,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--img", action="store_true")


    args = parser.parse_args()

    return args


def main(args):
    check_path = os.path.join(args.output_dir, args.mode, args.name)
    mkdir(check_path)
    log_path = os.path.join("tensorboard", git_name, args.mode, args.name)
    mkdir(log_path)
    writer = SummaryWriter(log_path)
    model_num_class = (
        {
            "dryness": 5,
            "pigmentation_forehead": 6,
            "pigmentation_cheek": 6,
            "pore": 6,
            "sagging": 7,
            "wrinkle_forehead": 7,
            "wrinkle_glabellus": 7,
            "wrinkle_perocular": 7
        }  # dryness, pigmentation, pore, sagging, wrinkle
        if args.mode == "class"
        else {
            "count": 1,
            "pore": 1,
            "wrinkle": 1,
            "elasticity": 1,
            "moisture": 1,
        }  # pigmentation, pore, wrinkle, elasticity, moisture
    )

    args.best_loss, model_list = dict(), dict()
    args.best_loss.update({item: np.inf for item in model_num_class})
    model_list.update({key: models.resnet50(weights=None, num_classes = value) for key, value in model_num_class.items()})

    ## Adjust the number of output in model for each region image
    model_dict_path = os.path.join(check_path, "pore", "state_dict.bin")
    args.save_img = os.path.join("save_img", git_name, args.mode, args.name)
    
    if args.reset:
        print(f"\033[90mReseting......{model_dict_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
        if os.path.isdir(args.save_img):
            shutil.rmtree(args.save_img)
    
    else:
        if os.path.isfile(model_dict_path):
            print(f"\033[92mResuming......{model_dict_path}\033[0m")

            for key, model in model_list.items():
                model_list[key] = resume_checkpoint(
                    args,
                    model,
                    os.path.join(check_path, f"{key}", "state_dict.bin"),
                )

    # for key, model in model_list.items():
    #     for name, param in model.named_parameters():
    #         if 'layer1' in name or 'layer2' in name or 'layer3' in name:
    #             param.requires_grad = False
            
    for key in model_list:
        # model_list[key] = torch.nn.DataParallel(model_list[key])
        model_list[key] = model_list[key].cuda()
            
    logger = setup_logger(args.name + args.mode, check_path)
    logger.info(args)
    logger.info("Command Line: " + " ".join(sys.argv))

    dataset = CustomDataset(args)

    trainset_loader = dataset.load_dataset("train")
    valset_loader = dataset.load_dataset("valid")
    
    
    resnet_model = Model(
        args, model_list, trainset_loader, valset_loader, logger, check_path, model_num_class, writer
    )

    for epoch in range(args.load_epoch, args.epoch):
        resnet_model.update_e(epoch + 1) if args.load_epoch else None
        resnet_model.run(phase = 'train')
        resnet_model.run(phase = 'valid')

        resnet_model.update_m(model_num_class)
        resnet_model.update_e(epoch + 1)
        resnet_model.reset_log(mode=args.mode)

        if resnet_model.stop_early():
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
