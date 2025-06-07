from collections import defaultdict
import json
import shutil
import sys
import os

import torch
import gc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import models
from tool.data_loader import CustomDataset_class, CustomDataset_regress
import argparse
from tool.logger import setup_logger
from torch.utils import data
import torch.nn as nn
from tool.model import Model_test
from tool.utils import resume_checkpoint, fix_seed


git_name = "None"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[2], choices=[1, 2, 3], nargs="+")

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )
    
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )
    
    args = parser.parse_args()

    return args


def main(args):
    args.root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args.git_name = git_name
    check_path = os.path.join(args.root_path , "checkpoint", git_name, args.mode, args.name)

    if os.path.isdir(os.path.join(check_path, "log", "eval")):
        shutil.rmtree(os.path.join(check_path, "log", "eval"))
        
    args.log_path = os.path.join(check_path, "log")
    logger = setup_logger(
        args.name,
        os.path.join(args.log_path, "eval")
    )
    logger.info("Command Line: " + " ".join(sys.argv))

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
        key: models.coatnet.coatnet_4(num_classes=value)
        for key, value in model_num_class.items()
    }

    dataset = (
        CustomDataset_class(args, logger, "test")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )

    model_area_dict = (
        {
            "dryness": ["dryness"],
            "pigmentation": ["forehead_pigmentation", "cheek_pigmentation"],
            "pore": ["pore"],
            "sagging": ["sagging"],
            "wrinkle": ["forehead_wrinkle", "glabellus_wrinkle", "perocular_wrinkle"],
        }
        if args.mode == "class"
        else {
            "pigmentation": ["pigmentation"],
            "moisture": ["moisture_forehead", "moisture_cheek", "moisture_chin"],
            "elasticity_R2": [
                "elasticity_R2_forehead",
                "elasticity_R2_cheek",
                "elasticity_R2_chin",
            ],
            "wrinkle_Ra": ["wrinkle_Ra_perocular"],
            "pore": ["pore_cheek"],
        }
    )
    
    test_dict = defaultdict(list)
    for key in model_list:
        for w_key in model_area_dict[key]:
            testset, _ = dataset.load_dataset("test", w_key)
            test_dict[w_key] = [i[2] for i in testset]
    
    check_path = "/home/jeongho/dir/NIA/dataset/split/smart_pad"
    mode = "w" 
    
    with open(f"{check_path}/{args.seed}_testset_info.txt", mode) as f:
        json.dump(test_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
