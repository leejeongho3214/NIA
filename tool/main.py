from datetime import datetime
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 스크립트 디렉토리 강제 설정
script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(script_dir)

import torch
import yaml

from torch.utils import data
import shutil
import numpy as np

from custom_model.coatnet import coatnet_4
from utils import mkdir, resume_checkpoint, fix_seed
from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
import argparse
from tool.model import Model
<<<<<<< HEAD
=======

import wandb  # 상단에 추가
>>>>>>> 6227645 (250505 Add a wandb code)

git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

<<<<<<< HEAD
    parser.add_argument("--equ", type=int, default=2, choices=[1, 2, 3])
=======
    parser.add_argument("--equ", type=int, default=1, choices=[1, 2, 3])
>>>>>>> 6227645 (250505 Add a wandb code)

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
        "--lr",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
    )
    
    parser.add_argument(
        "--num_gpu",
        default=1,
        type=int,
    )
    
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
    )


    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--ddp", action="store_true")

    args = parser.parse_args()
    
    args.num_workers = int(8 / args.num_gpu)

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
            key: coatnet_4(num_classes=value)
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
                model_list[path], info, global_step, args.run_id = resume_checkpoint(
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
        for code_name in ["tool/main.py", "tool/data_loader.py", "tool/model.py", "custom_model/resnet.py", "custom_model/coatnet.py"]]
    
    args_dict = vars(args)

    # YAML 파일에 저장
    yaml_file_path = os.path.join(code_path, 'config.yaml')
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file, default_flow_style=False)
    
    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info(f"[{git_name}]Command Line: " + " ".join(sys.argv))

    dataset = (
        CustomDataset_class(args, logger, "train")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )
    
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0  # 기본값 설정
    
    torch.cuda.set_device(args.local_rank)
     
    for key in model_list:
        if key in pass_list:
            continue
        
        if args.ddp: torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=args.num_gpu, rank=args.local_rank)
        
        if not loading: args.run_id = str(uuid.uuid4())  # 고유한 run id 생성
        # args.name으로 프로젝트 식별
        wandb_run = wandb.init(
            project = "NIA-Korean-Facial-Assessment",
            name = f"{now:%Y.%m.%d}/{args.git_name}/{args.name}_{key}",
            config = vars(args),
            resume = True if loading else False,
            id = args.run_id,
        )

        model = model_list[key].cuda()
        if args.ddp: model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        ) 

        trainset, grade_num = dataset.load_dataset("train", key)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=args.num_gpu, rank=args.local_rank, shuffle=True) if args.ddp else None
        trainset_loader = data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size // args.num_gpu,
            num_workers=args.num_workers,
            shuffle=False if args.ddp else True,
            sampler=train_sampler,
        )
        valset, _ = dataset.load_dataset("val", key)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=args.num_gpu, rank=args.local_rank, shuffle=False)  if args.ddp else None
        valset_loader = data.DataLoader(
            dataset=valset,
            batch_size=args.batch_size // args.num_gpu,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=val_sampler,
        )

        resnet_model = Model(
            args = args,
            model = model,
            temp_model = None,
            train_loader = trainset_loader,
            valid_loader = valset_loader,
            logger = logger,
            best_loss = args.best_loss,
            check_path = check_path,
            model_num_class = model_num_class,
            wandb_run = wandb_run,
            m_dig = key,
            grade_num = grade_num,
            info = info if loading else None, 
            global_step = global_step if loading else 0
        )

        if args.load_epoch[key] < 50:
            for epoch in range(args.load_epoch[key], args.epoch):
                if args.load_epoch[key]:
                    resnet_model.update_e(epoch + 1, *info) 
                    
                resnet_model.train()
                
                resnet_model.valid()
                resnet_model.reset_log(True)

        resnet_model.print_best()
        wandb.finish()



if __name__ == "__main__":
    args = parse_args()
    main(args)