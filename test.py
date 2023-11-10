import numpy as np
import torch
from torchvision import models

import os
import copy
from torch.utils.data import random_split
from data_loader import CustomDataset
from model import resume_checkpoint, mkdir
from test_model import Model_test
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import argparse
from torch.utils import data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="100%/1,2,3",
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

    parser.add_argument("--stop_early", type=int, default=30)

    parser.add_argument("--equ", type=int, default=[1, 2, 3], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--angle", default="F", type=str, choices=["F", "all"])

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
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()

    return args


def build_dataset(args, logger):
    train_dataset, val_dataset, test_dataset = random_split(
        CustomDataset(args),
        [0.8, 0.1, 0.1],
        generator=torch.Generator().manual_seed(523),
    )
    ## For consistent results, we have set a seed number

    return train_dataset, val_dataset, test_dataset


def main(args):
    log_path = os.path.join(args.loss_dir, args.mode, args.name)
    check_path = os.path.join(args.output_dir, args.mode, args.name)

    mkdir(log_path)
    mkdir(check_path)
    ## Make the directories for save


    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_list = [copy.deepcopy(model) for _ in range(9)]
    # Define 8 resnet models for each region

    model_num_class = (
        [np.nan, 15, 9, 9, 0, 12, 0, 5, 7]
        if args.mode == "class"
        else [1, 2, np.nan, 1, 0, 3, 0, np.nan, 2]
    )

    resume_list = list()
    for idx, item in enumerate(model_num_class):
        if not np.isnan(item):
            model_list[idx].fc = nn.Linear(
                model_list[idx].fc.in_features, model_num_class[idx]
            )
            resume_list.append(idx)

    ## Adjust the number of output in model for each region image

    model_dict_path = os.path.join(check_path, "1", "state_dict.bin")


    if os.path.isfile(model_dict_path):
        print(f"\033[92mResuming......{check_path}\033[0m")

        for idx in resume_list:
            if idx in [4, 6]: continue
            model_list[idx] = resume_checkpoint(
                args,
                model_list[idx],
                os.path.join(check_path, f"{idx}", "state_dict.bin"),
            )
    else:
        assert 0, "Check the check-point path, there's not any file in that"

    _, _, test_dataset = build_dataset(args, None)


    testset_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    # Data Loader

    resnet_model = Model_test(args, model_list, testset_loader)
    # If the model's acc is higher than best acc, it saves this model

    resnet_model.test(model_num_class, testset_loader)

    resnet_model.print_total()
    # Show the result for each value, such as pigmentation and pore, by averaging all of them

if __name__ == "__main__":
    args = parse_args()
    main(args)
