import torch
from torchvision import models

import sys
import os
from data_loader import CustomDataset
import argparse
from logger import setup_logger
from torch.utils import data

from model import Model_test

from utils import resume_checkpoint, fix_seed

fix_seed(523)
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

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--meta", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    args.check_path = os.path.join(args.output_dir, args.mode, args.name)
    ## Make the directories for save

    logger = setup_logger(
        args.name, os.path.join(args.check_path, "eval"), filename=args.name + ".txt"
    )
    logger.info("Command Line: " + " ".join(sys.argv))

    model_num_class = (
        {
            "dryness": 5,
            "pigmentation_forehead": 6,
            "pigmentation_cheek": 6,
            "pore": 6,
            "sagging": 7,
            "wrinkle_forehead": 7,
            "wrinkle_glabellus": 7,
            "wrinkle_perocular": 7,
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

    model_list = dict()
    model_list.update(
        {
            key: models.resnet50(weights=None, num_classes=value, args=args)
            for key, value in model_num_class.items()
        }
    )

    ## Adjust the number of output in model for each region image

    model_path = os.path.join(args.check_path, "save_model")
    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path] = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(dig_path, "state_dict.bin"),
                )

    dataset = CustomDataset(args)
    resnet_model = Model_test(args, logger)

    for key in model_list:
        model = model_list[key].cuda()
        testset = dataset.load_dataset("test", key)
        testset_loader = data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        resnet_model.test(model, testset_loader, key)
        resnet_model.print_test()
    resnet_model.save_value()


if __name__ == "__main__":
    args = parse_args()
    main(args)
