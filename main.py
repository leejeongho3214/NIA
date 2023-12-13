import os

import shutil
import sys
import numpy as np
from torchvision import models

from tensorboardX import SummaryWriter
import copy
from utils import mkdir
from logger import setup_logger
from data_loader import CustomDataset
from model import resume_checkpoint, Model
from torchvision.models import ResNet50_Weights
import torch.nn as nn

import argparse
from torch.utils import data


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
        "--loss_dir",
        default="tensorboard",
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
        default="checkpoint_new",
        type=str,
    )

    parser.add_argument(
        "--epoch",
        default=300,
        type=int,
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
        "--label_smooth",
        default=0,
        type=float,
    )

    parser.add_argument(
        "--lr",
        default=1e-3,
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

    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    log_path = os.path.join(args.loss_dir, args.mode, args.name)
    check_path = os.path.join(args.output_dir, args.mode, args.name)

    writer = SummaryWriter(log_path)
    mkdir(log_path)
    mkdir(check_path)
    ## Make the directories for save

    model = models.resnet50(weights=None)

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
    model_list.update({item: copy.deepcopy(model) for item in model_num_class})

    ## Adjust the number of output in model for each region image
    model_dict_path = os.path.join(check_path, "wrinkle", "state_dict.bin")

    if args.reset:
        print(f"\033[90mReseting......{model_dict_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
            mkdir(check_path)
    # If there is check-point, load that

    if os.path.isfile(model_dict_path):
        print(f"\033[92mResuming......{model_dict_path}\033[0m")

        for item in model_num_class:
            model_list[item] = resume_checkpoint(
                args,
                model_list[item],
                os.path.join(check_path, f"{item}", "state_dict.bin"),
            )

    logger = setup_logger(args.name + args.mode, check_path)
    logger.info(args)
    logger.info("Command Line: " + " ".join(sys.argv))

    dataset = CustomDataset(args)

    dataset.load_dataset(args, "train")
    trainset_loader = data.DataLoader(
        dataset=copy.deepcopy(dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )

    dataset.load_dataset(args, "val")
    valset_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    resnet_model = Model(
        args, model_list, trainset_loader, valset_loader, logger, writer, check_path
    )

    for epoch in range(args.load_epoch, args.epoch):
        resnet_model.update_e(epoch + 1) if args.load_epoch else None

        for model_idx in model_num_class:
            # In regression task, there are no images for 미간, 입술, 턱
            resnet_model.choice(model_idx)
            # Change the model for each region
            resnet_model.run(model_idx, phase="train")
            resnet_model.run(model_idx, phase="valid")

        resnet_model.update_m(model_num_class)
        resnet_model.save_value()
        # Show the result for each value, such as pigmentation and pore, by averaging all of them
        resnet_model.update_e(epoch + 1)
        resnet_model.reset_log(mode=args.mode)

        if resnet_model.stop_early():
            break

    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
