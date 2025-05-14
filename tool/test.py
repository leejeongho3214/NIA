import shutil
import sys
import os
import argparse

import yaml



from torch.utils.data import DataLoader
from tool.utils import resume_checkpoint, fix_seed
from tool.data_loader import CustomDataset_class, CustomDataset_regress
from tool.logger import setup_logger
from tool.model import Model_test
from custom_model.coatnet import coatnet_4

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
        "--res",
        default=256,
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
    
    seed = args.name.split("st")[0]
    if seed.isdigit():
        ValueError, f"It's not correct name, {args.name} -> {seed}"
    args.seed = int(seed)
    fix_seed(int(seed))
    
    args.git_name = git_name
    check_path = os.path.join("checkpoint", git_name, args.mode, args.name)

    if os.path.isdir(os.path.join(check_path, "log", "eval")):
        shutil.rmtree(os.path.join(check_path, "log", "eval"))
        
    args.log_path = os.path.join(check_path, "log")
    
    logger = setup_logger(
        args.name,
        os.path.join(args.log_path, "eval")
    )
    logger.info("Command Line: " + " ".join(sys.argv))
    
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
        key: coatnet_4(num_classes=value)
        for key, value in model_num_class.items()
    }

    model_path = os.path.join(check_path, "save_model")
    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path], _, _, _ = resume_checkpoint(
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
    resnet_model.save_value()

if __name__ == "__main__":
    args = parse_args()
    main(args)
