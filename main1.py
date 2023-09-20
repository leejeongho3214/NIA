import numpy as np
from torchvision import models

from tensorboardX import SummaryWriter
import os
import copy
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from logger import setup_logger
from data_loader import CustomDataset
from model import Model, resume_checkpoint, mkdir
from torchvision.models import ResNet50_Weights
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from torch.utils import data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "name",
        default="base",
        type=str,
    )

    parser.add_argument(
        "--img_path",
        default="img",
        type=str,
    )

    parser.add_argument(
        "--loss_dir",
        default="tensorboard",
        type=str,
    )

    parser.add_argument("--stop_early", type=int, default=30)

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--angle", default="F", type=str, choices=["F", "all"])

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--json_path",
        default="label",
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default="checkpoint",
        type=str,
    )

    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--double", action="store_true")

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
        default=8,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()

    return args


def build_dataset(args, logger):
    train_dataset, val_dataset = random_split(CustomDataset(args), [0.9, 0.1])
    logger.info(
        f"Train Dataset => {len(train_dataset)} // Valid Dataset => {len(val_dataset)}"
    )

    return train_dataset, val_dataset


def main(args):
    log_path = os.path.join(args.loss_dir, args.mode, args.name)
    check_path = os.path.join(args.output_dir, args.mode, args.name)

    writer = SummaryWriter(log_path)
    mkdir(log_path)
    mkdir(check_path)
    ## Make the directories for save

    args.best_loss = (
        [0 for _ in range(8)] if args.mode == "class" else [np.inf for _ in range(8)]
    )

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_list = [copy.deepcopy(model) for _ in range(8)]
    # Define 8 resnet models for each region

    model_num_class = (
        [15, 9, 9, 9, 12, 12, 5, 7]
        if args.mode == "class"
        else [15, np.nan, 8, 8, 16, 16, np.nan, 15]
    )
    resume_list = list()
    for idx, item in enumerate(model_num_class):
        if not np.isnan(item):
            model_list[idx].fc = nn.Linear(
                model_list[idx].fc.in_features, model_num_class[idx]
            )
            resume_list.append(idx)

        model_num_class = (
            [15, 9, 9, 9, 12, 12, 5, 7]
            if args.mode == "class"
            else [15, np.nan, 8, 8, 16, 16, np.nan, 15]
        )
    ## Adjust the number of output in model for each region image

    model_dict_path = os.path.join(check_path, "0", "state_dict.bin")

    if os.path.isfile(model_dict_path):
        print(f"\033[92mResuming......{model_dict_path}\033[0m")

        for idx in resume_list:
            model_list[idx] = resume_checkpoint(
                args,
                model_list[idx],
                os.path.join(check_path, f"{idx}", "state_dict.bin"),
            )
    if args.reset:
        if os.path.isfile(os.path.join(check_path, "log.txt")):
            os.remove(os.path.join(check_path, "log.txt"))
    # If there is check-point, load that

    logger = setup_logger(args.name, check_path)
    logger.info(args)

    train_dataset, val_dataset = build_dataset(args, logger)

    trainset_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valset_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    # Data Loader

    resnet_model = Model(
        args, model_list, trainset_loader, valset_loader, logger, writer
    )

    for epoch in range(args.load_epoch, args.epoch):
        resnet_model.update_e(epoch + 1) if args.load_epoch else None

        for model_idx in range(8):
            if np.isnan(model_num_class[model_idx]):
                continue
            # In regression task, there are no images for 미간, 입술, 턱

            resnet_model.choice(model_idx)
            # Change the model for each region
            resnet_model.train()
            resnet_model.valid()

        resnet_model.update_m(model_num_class)
        # If the model's acc is higher than best acc, it saves this model
        for model_idx in range(8):
            if np.isnan(model_num_class[model_idx]):
                continue
            resnet_model.choice(model_idx)
            resnet_model.test()

        resnet_model.print_total()
        # Show the result for each value, such as pigmentation and pore, by averaging all of them
        resnet_model.update_e(epoch + 1)
        resnet_model.reset_log()

        if resnet_model.stop_early():
            break
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
