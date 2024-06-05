import inspect
import os
import sys
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
from torch.utils import data
import shutil

import numpy as np
from torchvision import models
from tensorboardX import SummaryWriter
from utils import FocalLoss, mkdir, resume_checkpoint, fix_seed
from logger import setup_logger
from tool.data_loader import CustomDataset_class, CustomDataset_regress
from model import Model
import argparse

fix_seed(523)
if len(os.popen("git branch --show-current").readlines()):
    git_name = os.popen("git branch --show-current").readlines()[0].rstrip()
else:
    git_name = os.popen("git describe --tags").readlines()[0].rstrip()



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

    parser.add_argument("--stop_early", type=int, default=50)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--aug",
        default=None,
        nargs="+",
        choices=["jitter", "crop"],
        type=str,
    )

    parser.add_argument(
        "--pass_list",
        default=[],
        nargs="+",
        choices=[
            "dryness",
            "pigmentation_forehead",
            "pigmentation_cheek",
            "pore",
            "sagging",
            "wrinkle_forehead",
            "wrinkle_glabellus",
            "wrinkle_perocular",
            "pigmentation",
            "forehead_moisture",
            "forehead_elasticity_R2",
            "perocular_wrinkle_Ra",
            "cheek_moisture",
            "cheek_elasticity_R2",
            "cheek_pore",
            "chin_moisture",
            "chin_elasticity_R2",
        ],
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
        default=200,
        type=int,
    )

    parser.add_argument(
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--gamma",
        default=3,
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
        default=0.005,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--img", action="store_true")
    parser.add_argument("--meta", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    check_path = os.path.join(args.output_dir, args.mode, args.name)
    log_path = os.path.join("tensorboard", git_name, args.mode, args.name)

    args.model = "cnn"
    
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
    pass_list = list()

    args.best_loss, model_list = dict(), dict()
    args.best_loss.update({item: np.inf for item in model_num_class})

    if args.model != "coatnet":
        model_list.update(
            {
                key: models.resnet50(weights=None, num_classes=value, args=args)
                for key, value in model_num_class.items()
            }
        )
    else:
        model_list.update(
            {
                key: models.coatnet.coatnet_4(num_classes=value, bias_v=args.bias)
                for key, value in model_num_class.items()
            }
        )

    args.save_img = os.path.join(check_path, "save_img")
    args.pred_path = os.path.join(check_path, "prediction")
    model_path = os.path.join(check_path, "save_model")

    if args.reset:
        print(f"\033[90mReseting......{check_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
        if os.path.isdir(log_path):
            shutil.rmtree(log_path)

    else:
        if os.path.isdir(model_path):
            for path in os.listdir(model_path):
                dig_path = os.path.join(model_path, path)
                if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                    print(f"\033[92mResuming......{dig_path}\033[0m")
                    model_list[path] = resume_checkpoint(
                        args,
                        model_list[path],
                        os.path.join(model_path, f"{path}", "state_dict.bin"),
                    )
                    if os.path.isdir(os.path.join(dig_path, "done")):
                        print(f"\043[92mPassing......{dig_path}\043[0m")
                        pass_list.append(path)

    pass_list = pass_list + args.pass_list

    mkdir(model_path)
    mkdir(log_path)
    writer = SummaryWriter(log_path)

    logger = setup_logger(
        args.name + args.mode, os.path.join(check_path, "log", "train")
    )
    logger.info(args)
    logger.info("Command Line: " + " ".join(sys.argv))
    logger.debug(inspect.getsource(FocalLoss))
    logger.debug(inspect.getsource(models.resnet.ResNet._forward_impl))
    logger.debug(inspect.getsource(Model.train))
    
    dataset = (
        CustomDataset_class(args, logger, "train")
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )

    for key in model_list:
        if key in pass_list:
            continue

        model = model_list[key].cuda()

        trainset = dataset.load_dataset("train", key)
        trainset_loader = data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )

        valset = dataset.load_dataset("valid", key)
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
        )

        for epoch in range(args.load_epoch, args.epoch):
            resnet_model.update_e(epoch + 1) if args.load_epoch else None

            resnet_model.train()
            resnet_model.valid()

            resnet_model.update_e(epoch + 1)
            resnet_model.reset_log()

            if resnet_model.stop_early():
                break

        del trainset_loader, valset_loader

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    args = parse_args()
    main(args)
