from datetime import datetime
import os
import sys
import uuid
from utils import get_loader, load_checkpoint, fix_seed, save_code_copy, path_organize

path_organize()

import torch

import shutil
import numpy as np

from custom_model.coatnet import coatnet_4

from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
import argparse
from tool.model import Model

import wandb 

git_name = os.popen("git branch --show-current").readlines()[0].rstrip()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=1, choices=[1, 2, 3])

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
        "--stop_early",
        default=15,
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
    args.num_workers = 8

    return args


def main(args):
    now = datetime.now()
    fix_seed(args.seed)
    args.git_name = git_name

    check_path = os.path.join("checkpoint", git_name, args.mode, args.name)
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
        key: coatnet_4(num_classes=value) for key, value in model_num_class.items()
    }

    model_path = os.path.join(check_path, "save_model")
    args.pred_path = os.path.join(check_path, "prediction")

    if args.reset:
        print(f"\033[90mReseting......{check_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)

    loading = False
    loading, model_list, pass_list, info, global_step, run_id = load_checkpoint(
        args, model_path, model_list, pass_list, loading
    )

    save_code_copy(args, check_path, model_path)

    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info(f"[{git_name}]Command Line: " + " ".join(sys.argv))

    dataset = (
        CustomDataset_class(args, logger, "train")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )

    for key in model_list:
        if key in pass_list:
            continue

        if args.ddp:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

        if loading:
            args.run_id = run_id if run_id is not None else str(uuid.uuid4())
            loading = False
        else:
            args.run_id = str(uuid.uuid4())  # 고유한 run id 생성

        api = wandb.Api()

        project_path = "NIA-Korean-Facial-Assessment"
        target_name = f"{now:%Y.%m.%d}/{args.git_name}/{args.name}_{key}"

        runs = api.runs(project_path)

        # name이 일치하는 모든 run 찾기
        matched_runs = [run for run in runs if run.name == target_name]

        if matched_runs:
            for run in matched_runs:
                print(f"Found run with name: {run.name}")
                print(f"Run ID: {run.id}")
                print(f"Run State: {run.state}")
                args.run_id = run.id
        else:
            print(f"No run found with name: {target_name}")

        # args.name으로 프로젝트 식별
        wandb_run = wandb.init(
            project="NIA-Korean-Facial-Assessment",
            name=target_name,
            config=vars(args),
            resume=True if (loading or matched_runs) else False,
            id=args.run_id,
        )

        model = model_list[key].cuda()
        if args.ddp:
            model = torch.nn.parallel.DistributedDataParallel(model)

        trainset_loader, grade_num = get_loader(dataset, "train", key, args)
        valset_loader, _ = get_loader(dataset, "val", key, args)

        each_model = Model(
            args=args,
            model=model,
            temp_model=None,
            train_loader=trainset_loader,
            valid_loader=valset_loader,
            logger=logger,
            best_loss=args.best_loss,
            check_path=check_path,
            model_num_class=model_num_class,
            wandb_run=wandb_run,
            m_dig=key,
            grade_num=grade_num,
            info=info if loading else None,
            global_step=global_step if loading else 0,
        )

        for epoch in range(args.load_epoch[key], args.epoch):
            if args.load_epoch[key]:
                each_model.update_e(epoch + 1, **info)

            each_model.train()
            each_model.valid()
            each_model.reset_log()

            if each_model.stop_early():
                break

        each_model.print_best()
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)
