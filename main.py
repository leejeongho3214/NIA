import numpy as np
from torchvision import models
import torch
import json
import cv2
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, random_split
from matplotlib import pyplot as plt
from logger import setup_logger

import errno
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from torch.utils import data

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

folder_name = {
    "F": "01",
    "Fb": "07",
    "Ft": "06",
    "L15": "02",
    "L30": "03",
    "R15": "04",
    "R30": "05",
    "L": "02",
    "R": "03",
}

class_num_list = {
    "pigmentation": 6,
    "wrinkle": 9,
    "pore": 6,
    "dryness": 5,
    "sagging": 7,  ## 나중에 바꿔라
}

area_naming = {
    "0": "이마",
    "1": "미간",
    "2": "왼쪽눈가",
    "3": "오른쪽눈가",
    "4": "왼쪽볼",
    "5": "오른쪽볼",
    "6": "입술",
    "7": "턱",
}


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    return e_x / torch.sum(e_x)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="base",
        type=str,
    )

    parser.add_argument(
        "--img_path",
        default="img",
        type=str,
    )

    parser.add_argument("--equ", type=int, default=[1], choices=[1, 2, 3], nargs="+")

    parser.add_argument("--angle", default="F", choices=["F", "all"])

    parser.add_argument(
        "--json_path",
        default="label",
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default="checkpoint",
        type=str,
    )

    parser.add_argument("--normalize", action="store_true")

    parser.add_argument("--double", action="store_true")

    parser.add_argument(
        "--epoch",
        default=200,
        type=int,
    )
    parser.add_argument(
        "--load_epoch",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    return args


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_checkpoint(model, args, epoch, m_idx, best_loss):
    checkpoint_dir = os.path.join(args.output_dir, args.name, str(m_idx))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        os.path.join(checkpoint_dir, "state_dict.bin"),
    )

    return checkpoint_dir


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update_train(self, val, batch_size=32):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

    def update_val(self, val, num=1):
        self.val = val
        self.sum += val
        self.count += num
        self.avg = self.sum / self.count


class CustomDataset(Dataset):
    def __init__(self, args):
        self.img_path = args.img_path
        self.json_path = args.json_path
        sub_path_list = os.listdir(self.img_path)
        sub_path_list = [f"{equ:02}" for equ in args.equ]
        element = (
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
            if args.normalize
            else [
                transforms.ToTensor(),
            ]
        )

        self.transform = {
            "train": transforms.Compose(element),
            "test": transforms.Compose(element),
        }
        self.sub_path = list()
        for equ_name in sub_path_list:
            for sub_fold in os.listdir(os.path.join(self.img_path, equ_name)):
                if sub_fold.startswith("."):
                    continue
                img_count = 0
                for file_name in os.listdir(
                    os.path.join(self.img_path, equ_name, sub_fold)
                ):
                    if any(
                        file_name.lower().endswith(ext) for ext in [".jpg", ".jpeg"]
                    ):
                        img_count += 1
                if img_count == 7:
                    folder_path = os.path.join(self.img_path, equ_name, sub_fold)
                    for img_name in os.listdir(folder_path):
                        if not img_name.endswith((".jpg", ".jpeg")):  ## 일부러 F를 추가함
                            continue

                        if args.angle == "F":
                            if not img_name.endswith(("F.jpg")):
                                continue

                        img = cv2.imread(os.path.join(folder_path, img_name))

                        angle = img_name.split(".")[0].split("_")[-1]
                        area_list = list()
                        for idx_area in range(1, 9):
                            json_name = (
                                "_".join(img_name.split("_")[:2])
                                + f"_{angle}_{idx_area:02d}.json"
                            )
                            with open(
                                os.path.join(
                                    self.json_path,
                                    equ_name,
                                    folder_name[angle],
                                    json_name.split("_")[0],
                                    json_name,
                                ),
                                "r",
                                encoding="utf8",
                            ) as f:
                                meta = json.load(f)

                            bbox_point = meta["images"]["bbox"]
                            bbox_x = [
                                min(bbox_point[0], bbox_point[2]),
                                max(bbox_point[0], bbox_point[2]),
                            ]
                            bbox_y = [
                                min(bbox_point[1], bbox_point[3]),
                                max(bbox_point[1], bbox_point[3]),
                            ]

                            if (bbox_point[0] > bbox_point[2]) or (
                                bbox_point[1] > bbox_point[3]
                            ):
                                area_name = str(
                                    int(json_name.split("_")[-1].split(".")[0])
                                )
                                print(
                                    f"Sub No: {json_name.split('_')[0]} & Angle: {angle} & Area: {area_naming[area_name]}// 위치 반전"
                                )

                            patch_img = img[
                                bbox_y[0] : bbox_y[1], bbox_x[0] : bbox_x[1]
                            ]

                            reduction_value = max(patch_img.shape) / 128
                            try:
                                n_patch_img = cv2.resize(
                                    patch_img,
                                    (
                                        int(patch_img.shape[1] / reduction_value),
                                        int(patch_img.shape[0] / reduction_value),
                                    ),
                                )
                            except:
                                print(
                                    f"Sub No: {json_name.split('_')[0]} & Angle: {angle} & Area: {area_naming[area_name]} , w: {bbox_x[1]- bbox_x[0]}"
                                )
                                continue

                            patch_img = np.zeros([128, 128, 3], dtype=np.uint8)
                            if args.double:
                                if idx_area in [1, 7, 8]:
                                    if n_patch_img.shape[0] * 2 > 128:
                                        r_value = n_patch_img.shape[0] / 64
                                        n_patch_img = cv2.resize(
                                            n_patch_img,
                                            (
                                                int(n_patch_img.shape[1] / r_value),
                                                int(n_patch_img.shape[0] / r_value),
                                            ),
                                        )
                                    patch_img[
                                        : n_patch_img.shape[0], : n_patch_img.shape[1]
                                    ] = n_patch_img
                                    patch_img[
                                        n_patch_img.shape[0] : n_patch_img.shape[0] * 2,
                                        : n_patch_img.shape[1],
                                    ] = n_patch_img
                                elif idx_area in [3, 4]:
                                    if n_patch_img.shape[1] * 2 > 128:
                                        r_value = n_patch_img.shape[1] / 64
                                        n_patch_img = cv2.resize(
                                            n_patch_img,
                                            (
                                                int(n_patch_img.shape[1] / r_value),
                                                int(n_patch_img.shape[0] / r_value),
                                            ),
                                        )
                                    patch_img[
                                        : n_patch_img.shape[0], : n_patch_img.shape[1]
                                    ] = n_patch_img

                                    patch_img[
                                        : n_patch_img.shape[0],
                                        n_patch_img.shape[1] : n_patch_img.shape[1] * 2,
                                    ] = n_patch_img
                                    if idx_area == 3:
                                        plt.imshow(patch_img)
                                else:
                                    patch_img[
                                        : n_patch_img.shape[0], : n_patch_img.shape[1]
                                    ] = n_patch_img

                            else:
                                patch_img[
                                    : n_patch_img.shape[0], : n_patch_img.shape[1]
                                ] = n_patch_img

                            pil_img = Image.fromarray(patch_img)
                            patch_img = self.transform["train"](pil_img)
                            label_data = meta["annotations"]
                            area_list.append([patch_img, label_data])
                        self.sub_path.append(area_list)

                else:
                    print(f"{os.path.join(self.img_path, equ_name, sub_fold)}는 제외됨")

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        return self.sub_path[idx]


def build_dataset(args):
    train_dataset, val_dataset = random_split(CustomDataset(args), [0.9, 0.1])

    return train_dataset, val_dataset


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.epoch / 2.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


log_losses = [AverageMeter() for _ in range(8)]
log_acc_list = {
    "pigmentation": AverageMeter(),
    "wrinkle": AverageMeter(),
    "sagging": AverageMeter(),
    "pore": AverageMeter(),
    "dryness": AverageMeter(),
}


class Model(object):
    def __init__(self, args, model_list, train_loader, valid_loader, logger):
        super(Model, self).__init__()
        self.args = args
        self.loss = log_losses
        self.model_list = model_list
        self.temp_model_list = [None for _ in range(8)]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.best_acc = args.best_loss
        self.val_acc = log_losses
        self.logger = logger

        self.log_loss = log_losses
        self.log_acc = log_acc_list
        self.epoch = 0
        self.criterion = nn.CrossEntropyLoss()

        self.phase = None
        self.m_idx = 0
        self.model = None

    def choice(self, m_idx):
        self.model = self.model_list[m_idx]
        self.m_idx = m_idx

    def temp_save_m(self):
        self.temp_model_list[self.m_idx] = self.model

    def update_m(
        self,
    ):
        for idx in range(8):
            if self.best_acc[idx] < self.val_acc[idx].avg:
                self.best_acc[idx] = self.val_acc[idx].avg
                self.model_list[idx] = self.temp_model_list[idx]
            save_checkpoint(
                self.model_list[idx], self.args, self.epoch, idx, self.best_acc
            )

    def update_e(self, epoch):
        self.epoch = epoch

    def print_acc(self):
        print(
            f"[{self.phase}] pigmentation: {(self.log_acc['pigmentation'].avg * 100):.2f}% // wrinkle: {(self.log_acc['wrinkle'].avg * 100):.2f}% // sagging: {(self.log_acc['sagging'].avg * 100):.2f}% // pore: {(self.log_acc['pore'].avg * 100):.2f}% // dryness: {(self.log_acc['dryness'].avg * 100):.2f}%"
        )
        self.logger.debug(
            f"Epoch: {self.epoch} [{self.phase}] pigmentation: {(self.log_acc['pigmentation'].avg * 100):.2f}% // wrinkle: {(self.log_acc['wrinkle'].avg * 100):.2f}% // sagging: {(self.log_acc['sagging'].avg * 100):.2f}% // pore: {(self.log_acc['pore'].avg * 100):.2f}% // dryness: {(self.log_acc['dryness'].avg * 100):.2f}%"
        )

    def reset_log(self):
        self.log_acc = log_acc_list
        self.val_acc = log_losses
        self.log_loss = log_losses

    def train(self):
        self.model = self.model_list[self.m_idx]
        self.model.train()
        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        self.log_loss
        area_num = self.m_idx
        for iteration, patch_list in enumerate(self.train_loader):
            img, label = patch_list[area_num][0].to(device), patch_list[area_num][1]
            num_class = sum([class_num_list[name] for name in label])
            self.model.fc = nn.Linear(self.model.fc.in_features, num_class)
            adjust_learning_rate(optimizer, self.epoch, self.args)
            pred = self.model.to(device)(img)

            num = 0
            loss = 0
            for name in label:
                gt = F.one_hot(
                    label[name].type(torch.int64), num_classes=class_num_list[name]
                )
                pred_p = softmax(pred[:, num : num + class_num_list[name]])
                num += class_num_list[name]
                loss += self.criterion(
                    pred_p.type(torch.float), gt.type(torch.float).to(device)
                )

            self.log_loss[area_num].update_train(loss, batch_size=gt.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration == 0 and self.epoch == 0:
                first_row_img = torch.concat(
                    [patch_list[idx][0][0].permute(1, 2, 0) for idx in range(4)]
                )
                second_row_img = torch.concat(
                    [patch_list[idx][0][0].permute(1, 2, 0) for idx in range(4, 8)]
                )
                mkdir(f"vis/{self.args.name}/{area_num}")
                cv2.imwrite(
                    f"vis/{self.args.name}/{area_num}/epoch_{self.epoch}.jpg",
                    torch.concat([first_row_img, second_row_img], dim=1).numpy() * 256,
                )

            # if iteration == 0:
            #     mkdir(f"vis/{area_num}")
            #     cv2.imwrite(
            #         f"vis/{area_num}/epoch_{self.epoch}.jpg",
            #         torch.concat(
            #             [img.detach().cpu()[idx].permute(1, 2, 0) for idx in range(2)]
            #         ).numpy()
            #         * 256,
            #     )

            if iteration == len(self.train_loader) - 1:
                print(
                    f"\rEpoch: {self.epoch} [Train][{area_naming[f'{area_num}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {self.log_loss[area_num].avg}"
                )
                self.logger.debug(
                    f"Epoch: {self.epoch} [Train][{area_naming[f'{area_num}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {self.log_loss[area_num].avg}"
                )
            else:
                print(
                    f"\rEpoch: {self.epoch} [Train][{area_naming[f'{area_num}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {self.log_loss[area_num].avg}",
                    end="",
                )

    def valid(self):
        self.phase = "valid"
        self.model = self.model_list[self.m_idx]
        self.model.eval()
        with torch.no_grad():
            area_num = self.m_idx
            for iteration, patch_list in enumerate(self.valid_loader):
                img, label = patch_list[area_num][0].to(device), patch_list[area_num][1]

                gt = (
                    torch.tensor(np.array([label[value].numpy() for value in label]))
                    .permute(1, 0)
                    .to(device)
                )

                num_class = [class_num_list[name] for name in label]
                self.model.fc = nn.Linear(self.model.fc.in_features, sum(num_class))
                pred = self.model.to(device)(img)

                num = 0
                count = 0
                for idx, area_name in enumerate(label):
                    pred_l = torch.argmax(
                        pred[:, num : num + class_num_list[area_name]], dim=1
                    )
                    num += class_num_list[area_name]
                    self.log_acc[area_name].update_val(
                        (pred_l == gt[:, idx]).sum().item(),
                        pred_l.shape[0],
                    )
                    count += (pred_l == gt[:, idx]).sum().item()

                self.val_acc[self.m_idx].update_val(count, gt.shape[0] * gt.shape[1])

                if iteration == len(self.valid_loader) - 1:
                    print(
                        f"\rEpoch: {self.epoch} [Val][{area_naming[f'{area_num}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.log_acc[area_name].avg * 100):.2f}%"
                    )
                    self.logger.debug(
                        f"Epoch: {self.epoch} [Val][{area_naming[f'{area_num}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.log_acc[area_name].avg * 100):.2f}%"
                    )

                else:
                    print(
                        f"\rEpoch: {self.epoch} [Val][{area_naming[f'{area_num}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.log_acc[area_name].avg * 100):.2f}%",
                        end="",
                    )

    def test(self):
        self.phase = "test"
        self.model = self.model_list[self.m_idx]
        self.model.eval()
        with torch.no_grad():
            area_num = self.m_idx
            for iteration, patch_list in enumerate(self.valid_loader):
                img, label = patch_list[area_num][0].to(device), patch_list[area_num][1]

                gt = (
                    torch.tensor(np.array([label[value].numpy() for value in label]))
                    .permute(1, 0)
                    .to(device)
                )

                num_class = [class_num_list[name] for name in label]
                self.model.fc = nn.Linear(self.model.fc.in_features, sum(num_class))
                pred = self.model.to(device)(img)

                num = 0
                for idx, area_name in enumerate(label):
                    pred_l = torch.argmax(
                        pred[:, num : num + class_num_list[area_name]], dim=1
                    )
                    num += class_num_list[area_name]
                    self.log_acc[area_name].update_val(
                        (pred_l == gt[:, idx]).sum().item(),
                        pred_l.shape[0],
                    )


def resume_checkpoint(args, model, path):
    state_dict = torch.load(path)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state_dict"], strict=False)
    del state_dict
    args.load_epoch = epoch
    args.best_loss = best_loss

    return model


def main(args):
    mkdir(os.path.join(args.output_dir, args.name))
    logger = setup_logger(args.name, os.path.join(args.output_dir, args.name), 0)
    logger.info(args)

    args.best_loss = [0 for _ in range(8)]

    model = models.resnet50(weights=True)
    model_dict_path = os.path.join(args.output_dir, "0", "model.bin")
    if os.path.isfile(model_dict_path):
        model_list = [
            resume_checkpoint[
                args,
                model,
                os.path.join(args.output_dir, args.name, f"{idx}", "model.bin"),
            ]
            for idx in range(8)
        ]
    else:
        model_list = [model for _ in range(8)]

    train_dataset, val_dataset = build_dataset(args)

    trainset_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valset_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    resnet_model = Model(args, model_list, trainset_loader, valset_loader, logger)

    for epoch in range(args.load_epoch, args.epoch):
        for model_idx in range(8):
            resnet_model.choice(model_idx)
            resnet_model.train()
            resnet_model.reset_log()
            resnet_model.valid()
            resnet_model.temp_save_m()

        resnet_model.update_m()
        resnet_model.reset_log()
        for model_idx in range(8):
            resnet_model.choice(model_idx)
            resnet_model.test()

        resnet_model.print_acc()
        resnet_model.update_e(epoch + 1)
        resnet_model.reset_log()


if __name__ == "__main__":
    args = parse_args()
    main(args)
