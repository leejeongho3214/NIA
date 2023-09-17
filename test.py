import numpy as np
import torch
from torchvision import models

import os
import copy
from torch.utils.data import random_split
from data_loader import CustomDataset
from model import resume_checkpoint
from test_model import Model_test
from torchvision.models import ResNet50_Weights
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

    args = parser.parse_args()

    return args


def build_dataset(args, logger):
    train_dataset, val_dataset = random_split(
        CustomDataset(args), [0.9, 0.1], generator=torch.Generator().manual_seed(523)
    )

    if logger is not None:
        logger.info(
            f"Train Dataset => {len(train_dataset)} // Valid Dataset => {len(val_dataset)}"
        )

    return train_dataset, val_dataset
 

def main(args):
    check_path = os.path.join(args.output_dir, args.mode, args.name)

    model_num_class = (
        [15, 9, 9, 9, 12, 12, 5, 7]
        if args.mode == "class"
        else [4, np.nan, 3, 3, 5, 5, np.nan, 4]
    )
    args.best_loss = (
        [0 for _ in range(8)] if args.mode == "class" else [np.inf for _ in range(8)]
    )

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_list = [copy.deepcopy(model) for _ in range(8)]
    resume_list = list()
    for idx, item in enumerate(model_num_class):
        if not np.isnan(item):
            model_list[idx].fc = nn.Linear(
                model_list[idx].fc.in_features, model_num_class[idx]
            )
            resume_list.append(idx)
    model_dict_path = os.path.join(check_path, "0", "state_dict.bin")

    if os.path.isfile(model_dict_path):
        print("Resuming~")
        for idx in resume_list:
            model_list[idx] = resume_checkpoint(
                args,
                model_list[idx],
                os.path.join(check_path, f"{idx}", "state_dict.bin"),
            )

    else:
        assert 0, "Check the check-point path, there's not any file in that"

    _, val_dataset = build_dataset(args, None)

    valset_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    resnet_model = Model_test(args, model_list, valset_loader)

    for model_idx in range(8):
        if np.isnan(model_num_class[model_idx]):
            continue
        resnet_model.choice(model_idx)
        resnet_model.test()
    resnet_model.print_test()


if __name__ == "__main__":
    args = parse_args()
    main(args)
