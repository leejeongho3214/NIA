import torch
from torchvision import models

import sys
import os
import copy
from data_loader import CustomDataset
from model import resume_checkpoint
from test_model import Model_test
from torchvision.models import ResNet50_Weights
import torch.nn as nn
import argparse
from logger import setup_logger
from torch.utils import data
from utils import save_value

torch.manual_seed(523)
torch.cuda.manual_seed_all(523)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

    
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
        "--data",
        default="test",
        type=str,
    )
    
    parser.add_argument(
        "--data_num",
        default=-1,
        type=int,
    )      
    parser.add_argument(
            "--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+"
        )
    
    parser.add_argument(
        "--output_dir",
        default="checkpoint_new",
        type=str,
    )
    
    parser.add_argument("--train", action="store_true")
    
    parser.add_argument("--log", action="store_true")

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
        "--res",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--label_smooth",
        default=0.5,
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
    
    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    check_path = os.path.join(args.output_dir, args.mode, args.name)
    ## Make the directories for save

    logger = setup_logger(args.name, "eval/" + args.mode, filename=args.name + ".txt")
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
    
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model_list = dict()
    model_list.update({item: copy.deepcopy(model) for item in model_num_class})
    # Define 8 resnet models for each region

    ## Class Definition


    resume_list = list()
    for item in model_num_class:
        model_list[item].fc = nn.Linear(
            model_list[item].fc.in_features, model_num_class[item]
        )
        resume_list.append(item)

    ## Adjust the number of output in model for each region image
    model_dict_path = os.path.join(check_path, "wrinkle", "state_dict.bin")

    if os.path.isfile(model_dict_path):
        logger.info(f"\033[92mResuming......{check_path}\033[0m")

        for idx in resume_list:
            model_list[idx] = resume_checkpoint(
                args,
                model_list[idx],
                os.path.join(check_path, f"{idx}", "state_dict.bin"),
            )

    dataset = CustomDataset(args)
    
    dataset.load_dataset(args, "test")
    testset_loader = data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    ) 
    # Data Loader

    resnet_model = Model_test(args, model_list, testset_loader, logger)
    # If the model's acc is higher than best acc, it saves this model
    logger.info("Inferece ...")
    resnet_model.test(model_num_class, testset_loader)
    logger.info("Finish!")
    resnet_model.save_value()

    return logger


if __name__ == "__main__":
    args = parse_args()
    logger = main(args)
