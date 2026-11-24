import sys
import os
import argparse
import yaml

workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, workspace_path)
os.chdir(workspace_path)

sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import torch
from torch.utils.data import DataLoader
from tool.utils import fix_seed
from tool.data_loader import CustomDataset
from tool.logger import setup_logger
from tool.model import Model_test
from torchvision import models

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="none",
        type=str,
    )

    parser.add_argument("--equ", type=int, nargs="+", default=[1], choices=[1, 2, 3])

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--check_root",
        default="",
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
        "--res",
        default=256,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
    )

    parser.add_argument(
        "--argmax",
        action="store_true",
        help="Class mode: score the argmax grade instead of the expected grade (paper uses the expected grade).",
    )

    args = parser.parse_args()

    return args


def load_weights(model, path):
    """Load a checkpoint saved by tool/main.py ({'model_state': ...}) or a plain state_dict."""
    try:  # weights-only files load without unpickling arbitrary objects
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # full training checkpoints also hold optimizer/scheduler/numpy state
        state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=True)
    return model


def main(args):
    seed = args.name.split("st")[0]
    if not seed.isdigit():
        raise ValueError(
            f"Incorrect name: {args.name}. Expected a run name starting with the split seed, e.g. '1st_resnet50'."
        )

    args.seed = int(seed)
    fix_seed(int(seed))

    check_path = os.path.join(
        args.check_root, "checkpoint", args.mode, args.name
    )

    model_path = os.path.join(check_path, "save_model")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Incorrect checkpoint path: {model_path} does not exist")

    args.log_path = os.path.join(check_path, "log")

    logger = setup_logger(args.name, os.path.join(args.log_path, "eval"))
    logger.info("Command Line: " + " ".join(sys.argv))

    os.makedirs(os.path.join(check_path, "code"), exist_ok=True)
    yaml_file_path = os.path.join(check_path, "code", "test_config.yaml")
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)

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
        key: models.resnet50(num_classes=value)
        for key, value in model_num_class.items()
    }

    for key in model_list:
        ckpt = os.path.join(model_path, key, "state_dict.bin")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        print(f"\033[92mLoading......{ckpt}\033[0m")
        model_list[key] = load_weights(model_list[key], ckpt)

    dataset = (
        CustomDataset(args, logger, "test", special=True)
        if args.mode == "class"
        else CustomDataset(args, logger, "test", special=True)
    )
    resnet_model = Model_test(args, logger)

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

    for key in model_list:
        model = model_list[key].cuda()
        for w_key in model_area_dict[key]:
            testset, _ = dataset.load_dataset(w_key)
            testset_loader = DataLoader(
                dataset=testset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
            )
            resnet_model.test(model, testset_loader, w_key)
            resnet_model.print_test()
    resnet_model.print_pooled()
    resnet_model.save_value()


if __name__ == "__main__":
    args = parse_args()
    main(args)
