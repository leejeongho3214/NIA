import torch
from torchvision import models

import sys
import os
from data_loader import CustomDataset
from test_model import Model_test
import argparse
from logger import setup_logger
from torch.utils import data

from utils import resume_checkpoint

torch.manual_seed(523)
torch.cuda.manual_seed_all(523)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cross", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    args.save_path = os.path.join(git_name, args.mode, args.name)
    check_path = os.path.join("checkpoint", args.save_path)
    ## Make the directories for save

    logger = setup_logger(args.name, os.path.join("eval", args.save_path), filename=args.name + ".txt")
    logger.info("Command Line: " + " ".join(sys.argv))
    
    model_num_class = (
        {'dryness': 5, 
         'pigmentation': 6,
         'pore': 6,
         'sagging': 7,
         'wrinkle': 7}         # dryness, pigmentation, pore, sagging, wrinkle
        if args.mode == "class"
        else {
            'count': 1,
            'pore': 1,
            'wrinkle': 1,
            'elasticity': 1,
            'moisture': 1}    # pigmentation, pore, wrinkle, elasticity, moisture
    )
    
    model_list = dict()
    model_list.update({key: models.resnet50(weights=None, num_classes = value) for key, value in model_num_class.items()})
    # Define 8 resnet models for each region


    ## Adjust the number of output in model for each region image
    model_dict_path = os.path.join(check_path, "wrinkle", "state_dict.bin")

    if os.path.isfile(model_dict_path):
        logger.info(f"\033[92mResuming......{check_path}\033[0m")

        for key, model in model_list.items():
            model_list[key] = resume_checkpoint(
                args,
                model,
                os.path.join(check_path, f"{key}", "state_dict.bin"),
            )

    dataset = CustomDataset(args)
    testset_loader = dataset.load_dataset(args, "test")

    resnet_model = Model_test(args, model_list, testset_loader, logger)
    # If the model's acc is higher than best acc, it saves this model
    logger.info("Inferece ...")
    resnet_model.test()
    logger.info("Finish!")
    resnet_model.save_value()

    return logger


if __name__ == "__main__":
    args = parse_args()
    logger = main(args)
