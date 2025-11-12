from datetime import datetime
import os
import sys
import uuid

workspace_path = os.path.join(os.path.expanduser("~"), "dir/NIA")
sys.path.insert(0, workspace_path)
os.chdir(workspace_path)

sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

from utils import load_checkpoint, fix_seed, save_code_copy
import torch
import shutil
import numpy as np

from custom_model.coatnet import coatnet_1, coatnet_2, coatnet_3, coatnet_4

from logger import setup_logger
from tool.data_loader import CustomDataset
import argparse
from torch.utils.data import WeightedRandomSampler

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

    parser.add_argument(
        "--equ",
        type=int,
        nargs="+",
        default=[1],
        choices=[1, 2, 3],
    )

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
        "--coatnet",
        default=4, 
        type=int,
    )

    parser.add_argument(
        "--warmup_epochs",
        default=5,
        type=int,
    )

    parser.add_argument(
        "--warmup_ratio",
        default=0.1,
        type=float,
        help="Fallback ratio of total epochs to use for warmup when warmup_epochs <= 0",
    )

    parser.add_argument(
        "--lr_min_scale",
        default=0.01,
        type=float,
    )

    parser.add_argument(
        "--lr_min",
        default=None,
        type=float,
        help="Absolute minimum learning rate; defaults to lr * lr_min_scale",
    )

    parser.add_argument(
        "--lr_scheduler",
        default="cosine",
        choices=["cosine", "multistep"],
        type=str,
    )

    parser.add_argument(
        "--decay_milestones",
        type=int,
        nargs="+",
        default=[5, 10, 15],
        help="Epochs (0-indexed) at which to decay LR for multistep scheduler",
    )

    parser.add_argument(
        "--decay_gamma",
        default=0.5,
        type=float,
        help="Multiplicative gamma applied at each milestone for multistep scheduler",
    )

    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--aug_level",
        default="light",
        choices=["none", "light", "medium", "heavy"],
        help="Controls strength of train-time augmentations applied to cropped facial patches.",
    )

    parser.add_argument(
        "--allow_hflip",
        action="store_true",
        help="Enable horizontal flips during augmentation (disable if left/right areas differ).",
    )

    parser.add_argument(
        "--gamma",
        default=2,
        type=float,
    )

    parser.add_argument(
        "--lr",
        default=0.0001,
        type=float,
    )

    parser.add_argument(
        "--patience",
        default=5,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--stop_early",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--grad_clip",
        default=1.0,
        type=float,
    )

    parser.add_argument("--turn_epoch", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--ddp", action="store_true")

    args = parser.parse_args()
    args.num_workers = 8

    args.decay_milestones = sorted({m for m in args.decay_milestones if m >= 0})
    args.decay_gamma = max(0.0, min(1.0, args.decay_gamma))
    args.warmup_ratio = max(0.0, min(0.5, args.warmup_ratio))

    if args.warmup_epochs <= 0:
        args.warmup_epochs = max(1, int(args.epoch * args.warmup_ratio))
    args.warmup_epochs = min(args.warmup_epochs, max(1, args.epoch - 1))

    if args.lr_min is None:
        args.lr_min = max(1e-8, args.lr * args.lr_min_scale)
    else:
        args.lr_min = max(1e-8, args.lr_min)
        args.lr_min_scale = args.lr_min / max(args.lr, 1e-12)

    return args


def main(args):
    if args.coatnet < 3:
        args.batch_size = 32
    elif args.coatnet == 3:
        args.batch_size = 16
    else:
        args.batch_size = 8

    seed = args.name.split("st")[0]
    if seed.isdigit():
        ValueError, f"It's not correct name, {args.name} -> {seed}"

    fix_seed(int(seed))
    args.git_name = git_name
    args.seed = int(seed)

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

    model_choice = {1: coatnet_1, 2: coatnet_2, 3: coatnet_3, 4: coatnet_4}

    model_list = {
        key: model_choice[args.coatnet](num_classes=value) for key, value in model_num_class.items()
    }

    model_path = os.path.join(check_path, "save_model")
    args.pred_path = os.path.join(check_path, "prediction")

    if args.reset:
        print(f"\033[90mReseting......{check_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)

    loading = False
    if os.path.isdir(os.path.join(model_path, "pigmentation")):
        loading, model_list, pass_list, info, global_step, run_id = load_checkpoint(
            args, model_path, model_list, pass_list, loading
        )

    save_code_copy(args, check_path, model_path)

    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info(f"[{git_name}]Command Line: " + " ".join(sys.argv))

    train_dataset = (
        CustomDataset(args, logger, mode="train")
        if args.mode == "class"
        else CustomDataset(args, logger, mode="train")
    )

    val_dataset = (
        CustomDataset(args, logger, mode="val")
        if args.mode == "class"
        else CustomDataset(args, logger, mode="val")
    )

    test_dataset = (
        CustomDataset(args, logger, "test")
        if args.mode == "class"
        else CustomDataset(args, logger, "test")
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

        target_name = f"{args.git_name}/{args.name}_{key}"

        wandb_run = wandb.init(
           project="NIA-Korean-Facial-Assessment",
           name=target_name,
           config=vars(args),
        )

        model = model_list[key].cuda()
        if args.ddp:
            model = torch.nn.parallel.DistributedDataParallel(model)

        train_data, _ = train_dataset.load_dataset(key)
        val_data, _ = val_dataset.load_dataset(key)

        merged_data = train_data + val_data

        num_classes = model_num_class[key]
        class_counts = [0] * num_classes

        for sample in merged_data:
            label_value = int(sample[1])
            if label_value < 0 or label_value >= num_classes:
                raise ValueError(
                    f"Label value {label_value} for {key} out of range [0, {num_classes - 1}]"
                )
            class_counts[label_value] += 1

        grade_num = [max(1, count) for count in class_counts]

        class_weights = np.asarray(grade_num, dtype=np.float32)
        class_weights = 1.0 / np.sqrt(class_weights)
        class_weights /= np.sum(class_weights)
        class_weights = np.nan_to_num(class_weights, nan=0.0, posinf=0.0, neginf=0.0)

        sample_weights = [class_weights[int(sample[1])] for sample in merged_data]
        sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(merged_data),
            replacement=True,
        )

        trainset_loader = torch.utils.data.DataLoader(
            dataset=merged_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            sampler=sampler,
        )

        test_data, _ = test_dataset.load_dataset(key)
        testset_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        each_model = Model(
            args=args,
            model=model,
            temp_model=None,
            train_loader=trainset_loader,
            valid_loader=testset_loader,
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

            if epoch == 10 and args.turn_epoch:
                trainset_loader = torch.utils.data.DataLoader(
                    dataset=merged_data,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True,
                )
                each_model.train_loader = trainset_loader

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
