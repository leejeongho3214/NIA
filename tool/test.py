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


git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
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
    fix_seed(args.seed)
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
        {"dryness": 5, "pigmentation": 6, "pore": 6, "sagging": 6, "wrinkle": 7}
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
        key: models.resnet50(weights=models.ResNet50_Weights.DEFAULT, args=args)
        for key, _ in model_num_class.items()
    }

    for key, model in model_list.items(): 
        model.fc = nn.Linear(model.fc.in_features, model_num_class[key], bias = True)
        model_list.update({key: model})

    model_path = os.path.join(check_path, "save_model")
    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path], _ = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(dig_path, "state_dict.bin"),
                    path,
                    False,
                )
    else: 
        shutil.rmtree(check_path)
        assert 0, "Incorrect checkpoint path"

    dataset = (
        CustomDataset_class(args, logger, "test")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )
    resnet_model = Model_test(args, logger)

    model_area_dict = (
        {
            "dryness": ["dryness"],
            "pigmentation": ["pigmentation_forehead", "pigmentation_cheek"],
            "pore": ["pore"],
            "sagging": ["sagging"],
            "wrinkle": ["wrinkle_forehead", "wrinkle_glabellus", "wrinkle_perocular"],
        }
        if args.mode == "class"
        else {
            "pigmentation": ["pigmentation"],
            "moisture": ["forehead_moisture", "cheek_moisture", "chin_moisture"],
            "elasticity_R2": [
                "forehead_elasticity_R2",
                "cheek_elasticity_R2",
                "chin_elasticity_R2",
            ],
            "wrinkle_Ra": ["perocular_wrinkle_Ra"],
            "pore": ["cheek_pore"],
        }
    )
        
    print_dict = defaultdict(list)
    for key in model_list:
        model = model_list[key].cuda()
        for w_key in model_area_dict[key]:
            testset, _ = dataset.load_dataset("test", w_key)
            print_dict[w_key] = [i[2] for i in testset]
            testset_loader = data.DataLoader(
                dataset=testset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            resnet_model.test(model, testset_loader, w_key)
            resnet_model.print_test()
    resnet_model.save_value()
        
    with open(f"{check_path}/{args.seed}_testset_info.txt", "w") as f:
        json.dump(print_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)
