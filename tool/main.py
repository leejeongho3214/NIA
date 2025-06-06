from collections import defaultdict
import inspect
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
from torch.utils import data
import shutil
import torch.nn as nn
import numpy as np
from torchvision import models
from utils import mkdir, resume_checkpoint, fix_seed
from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
from model import Model
import argparse


git_name = "None"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[2], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--stop_early", type=int, default=50)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )


    parser.add_argument(
        "--epoch",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--gamma",
        default=2,
        type=float,
    )

    parser.add_argument(
        "--load_epoch",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )
    
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )
    
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
    )


    parser.add_argument("--reset", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    fix_seed(args.seed)
    args.root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args.git_name = git_name
    check_path = os.path.join(args.root_path , "checkpoint", git_name, args.mode, args.name)
    log_path = os.path.join(args.root_path , "tensorboard", git_name, args.mode, args.name)
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
    pass_list = list()

    args.best_loss = {item: np.inf for item in model_num_class}
    args.load_epoch = {item: 0 for item in model_num_class}

    model_list = {
            key: models.coatnet.coatnet_4(num_classes=value)
            for key, value in model_num_class.items()
        }
    
    model_path = os.path.join(check_path, "save_model")
        
    # for key, model in model_list.items(): 
    #     model.fc = nn.Linear(model.fc.in_features, model_num_class[key])
    #     model_list.update({key: model})

    args.save_img = os.path.join(check_path, "save_img")
    args.pred_path = os.path.join(check_path, "prediction")

    if args.reset:
        print(f"\033[90mReseting......{check_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)

    loading = False
    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path], info = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(model_path, f"{path}", "state_dict.bin"),
                    path, 
                )
                loading = True
                if os.path.isdir(os.path.join(dig_path, "done")):
                    print(f"\043[92mPassing......{dig_path}\043[0m")
                    pass_list.append(path)
            

    mkdir(model_path)
    mkdir(log_path)
    code_path = os.path.join(check_path, "code")
    mkdir(code_path)
    
    [shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), code_name), os.path.join(code_path, code_name.split("/")[-1])) \
        for code_name in ["main.py", "data_loader.py", "model.py", "../torchvision/models/resnet.py"]]
    
    args_dict = vars(args)

    # YAML 파일에 저장
    yaml_file_path = os.path.join(code_path, 'config.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)
    

    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info("Command Line: " + " ".join(sys.argv))

    dataset = (
        CustomDataset_class(args, logger, "train")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )
    train_dict = defaultdict(list)
    val_dict = defaultdict(list)
    test_dict = defaultdict(list)
    
    for key in model_list:
        
        trainset, grade_num = dataset.load_dataset("train", key)
        valset, _ = dataset.load_dataset("val", key)

        train_dict[key] = [i[2] for i in trainset]
        val_dict[key] = [i[2] for i in valset]
        
        check_path = "/home/jeongho/dir/NIA/dataset/split/smart_pad"
        
    mode = "w" 
    
    with open(f"{check_path}/{args.seed}_trainset_info.txt", mode) as f:
        json.dump(train_dict, f)
    with open(f"{check_path}/{args.seed}_valset_info.txt", mode) as f:
        json.dump(val_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)

    