import inspect
import os

import shutil
import sys
import numpy as np
from torchvision import models

import copy
from utils import mkdir, resume_checkpoint, labeling, LabelSmoothingCrossEntropy
from logger import setup_logger
from data_loader import CustomDataset
from model import Model

import argparse
from torch.utils import data

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
        default=1e-2,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
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
    ## Make the directories for save

    model_num_class = (
        {
            "dryness": 5,
            "pigmentation": 6,
            "pore": 6,
            "sagging": 7,
            "wrinkle": 7,
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
    model_dict_path = os.path.join(check_path, "wrinkle", "state_dict.bin")

    if args.reset:
        print(f"\033[90mReseting......{model_dict_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
            
    args.save_img = os.path.join("save_img", git_name, args.mode, args.name)
    if os.path.isdir(args.save_img):
        shutil.rmtree(args.save_img)
    # If there is check-point, load that
    
    else:
        if os.path.isfile(model_dict_path):
            print(f"\033[92mResuming......{model_dict_path}\033[0m")

            for key, model in model_list.items():
                model_list[key] = resume_checkpoint(
                    args,
                    model,
                    os.path.join(check_path, f"{key}", "state_dict.bin"),
                )

    logger = setup_logger(args.name + args.mode, check_path)
    logger.info(args)
    logger.info("Command Line: " + " ".join(sys.argv))
    logger.debug(inspect.getsource(labeling) if args.smooth else inspect.getsource(LabelSmoothingCrossEntropy))

    dataset = CustomDataset(args)

    dataset.load_dataset(args, "train")
    trainset_loader = data.DataLoader(
        dataset=copy.deepcopy(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    dataset.load_dataset(args, "valid")
    valset_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    resnet_model = Model(
        args, model_list, trainset_loader, valset_loader, logger, check_path, model_num_class
    )

    for epoch in range(args.load_epoch, args.epoch):
        resnet_model.update_e(epoch + 1) if args.load_epoch else None
        resnet_model.run(phase = 'train')
        resnet_model.run(phase = 'valid')

        resnet_model.update_m(model_num_class)
        resnet_model.save_value()

        resnet_model.update_e(epoch + 1)
        resnet_model.reset_log(mode=args.mode)

        if resnet_model.stop_early():
            break


if __name__ == "__main__":
    args = parse_args()
    main(args)
