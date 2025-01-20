from collections import defaultdict
import inspect
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 스크립트 디렉토리 강제 설정
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(script_dir)

from torch.utils import data
import shutil
import numpy as np
from torchvision import models
from tensorboardX import SummaryWriter
from utils import mkdir, resume_checkpoint, fix_seed
from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
from model import Model
import argparse

git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--stop_early", type=int, default=10)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )


    parser.add_argument(
        "--epoch",
        default=50,
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
        default=0.0001,
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
    args.git_name = git_name
    check_path = os.path.join("checkpoint", git_name, args.mode, args.name)
    log_path = os.path.join("tensorboard", git_name, args.mode, args.name)
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
    writer = SummaryWriter(log_path)
 
    [shutil.copy(os.path.join(os.getcwd(), code_name), os.path.join(code_path, code_name.split("/")[-1])) \
        for code_name in ["tool/main.py", "tool/data_loader.py", "tool/model.py", "torchvision/models/resnet.py"]]
    
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
    
    for key in model_list:
        if key in pass_list:
            continue
        
        model = model_list[key].cuda()
        trainset, grade_num = dataset.load_dataset("train", key)
        trainset_loader = data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        valset, _ = dataset.load_dataset("val", key)
        valset_loader = data.DataLoader(
            dataset=valset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        resnet_model = Model(
            args,
            model,
            trainset_loader,
            valset_loader,
            logger,
            check_path,
            model_num_class,
            writer,
            key,
            grade_num,
            info if loading else None, 
        )
        
        train_dict[key] = [i[2] for i in trainset]
        val_dict[key] = [i[2] for i in valset]
        

        if args.load_epoch[key] < 50:
            for epoch in range(args.load_epoch[key], args.epoch):
                if args.load_epoch[key]:
                    resnet_model.update_e(epoch + 1, *info) 
                    
                _ = resnet_model.train()
                
                resnet_model.valid()
                resnet_model.reset_log(True)

                if resnet_model.stop_early():
                    break
                
            resnet_model.print_best()
        del trainset_loader, valset_loader
        
        mode = "w" if os.path.isfile(f"{check_path}/log/train/trainset_info.txt") else "a"
        
        with open(f"{check_path}/log/train/trainset_info.txt", mode) as f:
            json.dump(train_dict, f)
        with open(f"{check_path}/log/train/valset_info.txt", mode) as f:
            json.dump(val_dict, f)


if __name__ == "__main__":
    args = parse_args()
    main(args)