import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import models
from tool.data_loader import CustomDataset_class, CustomDataset_regress
import argparse
from tool.logger import setup_logger
from torch.utils import data

from tool.model import Model_test
from tool.utils import resume_checkpoint, fix_seed

fix_seed(523)
git_name = os.popen("git branch --show-current").readlines()[0].rstrip()


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
    
    parser.add_argument(
        "--load_name",
        default = None,
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--stop_early", type=int, default=30)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
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
        "--aug",
        default=None,
        nargs="+",
        choices=["jitter", "crop"],
        type=str,
    )

    parser.add_argument(
        "--epoch",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--smooth",
        default=0.1,
        type=float,
    )

    parser.add_argument(
        "--res",
        default=128,
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
        default=1e-3,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--meta", action="store_true")

    args = parser.parse_args()

    return args


def main(args):
    args.check_path = os.path.join(args.output_dir, args.mode, args.name)
    ## Make the directories for save

    logger = setup_logger(
        args.name,
        os.path.join(args.check_path, "log", "eval"),
        filename=args.name + ".txt",
    )
    logger.info("Command Line: " + " ".join(sys.argv))

    model_num_class = (
        {
            "dryness": 5,
            "pigmentation_forehead": 6,
            "pigmentation_cheek": 6,
            "pore": 6,
            "sagging": 7,
            "wrinkle_forehead": 7,
            "wrinkle_glabellus": 7,
            "wrinkle_perocular": 7,
        }  # dryness, pigmentation, pore, sagging, wrinkle
        if args.mode == "class"
        else {
            "pigmentation": 1,
            "forehead_moisture": 1,
            "forehead_elasticity_R2": 1,
            "perocular_wrinkle_Ra": 1,
            "cheek_moisture": 1,
            "cheek_elasticity_R2": 1,
            "cheek_pore": 1,
            "chin_moisture": 1,
            "chin_elasticity_R2": 1,
        }
    )

    model_list = dict()
    model_list.update(
        {
            key: models.resnet50(weights=None, num_classes=value, args=args)
            for key, value in model_num_class.items()
        }
    )

    ## Adjust the number of output in model for each region image

    if args.load_name == None:
        args.load_name = args.name
    
    model_path = os.path.join(os.path.join(args.output_dir, args.mode, args.load_name), "save_model")
    if os.path.isdir(model_path):
        for path in os.listdir(model_path):
            dig_path = os.path.join(model_path, path)
            if os.path.isfile(os.path.join(dig_path, "state_dict.bin")):
                print(f"\033[92mResuming......{dig_path}\033[0m")
                model_list[path] = resume_checkpoint(
                    args,
                    model_list[path],
                    os.path.join(dig_path, "state_dict.bin"),
                )

    dataset = (
        CustomDataset_class(args, logger)
        if args.mode == "class"
        else CustomDataset_regress(args, logger)
    )
    resnet_model = Model_test(args, logger)

    for key in model_list:
        model = model_list[key].cuda()
        testset = dataset.load_dataset("test", key)
        testset_loader = data.DataLoader(
            dataset=testset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        resnet_model.test(model, testset_loader, key)
        resnet_model.print_test()
    resnet_model.save_value()


if __name__ == "__main__":
    args = parse_args()
    main(args)
