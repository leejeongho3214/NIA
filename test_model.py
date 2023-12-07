from collections import defaultdict
import datetime
import errno
import os
import cv2
import torch
import cv2
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import class_num_list, area_naming


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=32):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

    def update_acc(self, val, num=1):
        self.val = val
        self.sum += val
        self.count += num
        self.avg = self.sum / self.count


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    return e_x / torch.sum(e_x, dim=1).unsqueeze(dim=1)


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


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.epoch / 2.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model, args, epoch, m_idx, best_loss):
    checkpoint_dir = os.path.join(args.output_dir, args.mode, args.name, str(m_idx))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        os.path.join(checkpoint_dir, "temp_file.bin"),
    )

    os.rename(
        os.path.join(checkpoint_dir, "temp_file.bin"),
        os.path.join(checkpoint_dir, "state_dict.bin"),
    )
    return checkpoint_dir


def resume_checkpoint(args, model, path):
    state_dict = torch.load(path)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict
    args.load_epoch = epoch

    args.best_loss = best_loss

    return model


class Model_test(object):
    def __init__(self, args, model_list, testset_loader, logger):
        super(Model_test, self).__init__()
        self.args = args
        self.model_list = model_list
        self.logger = logger
        self.test_loader = testset_loader
        self.count = defaultdict(int)

        self.test_class_acc = {
            "sagging": AverageMeter(),
            "wrinkle": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation": AverageMeter(),
            "dryness": AverageMeter(),
        }
        self.test_regresion_mae = {
            "moisture": AverageMeter(),
            "wrinkle": AverageMeter(),
            "elasticity": AverageMeter(),
            "pore": AverageMeter(),
            "count": AverageMeter(),
        }

        self.equip_loss = {
            "0": {"count": 1},
            "1": {"moisture": 1, "elasticity": 1},
            "3": {"wrinkle": 1},
            "4": {"wrinkle": 1},
            "5": {"moisture": 1, "elasticity": 1, "pore": 1},
            "6": {"moisture": 1, "elasticity": 1, "pore": 1},
            "8": {"moisture": 1, "elasticity": 1},
        }
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0

    def choice(self, m_idx):
        if m_idx in [4, 6]:
            m_idx -= 1
            self.flag = True
        else:
            self.flag = False

        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def acc_avg(self, name):
        return round(self.test_class_acc[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_regresion_mae[name].avg, 4)

    def print_total(self, iter):
        if self.args.mode == "class":
            self.logger.info(
                f"pigmentation: {self.acc_avg('pigmentation')}%(T: {self.test_class_acc['pigmentation'].sum} / F: {self.test_class_acc['pigmentation'].count - self.test_class_acc['pigmentation'].sum}) // wrinkle: {self.acc_avg('wrinkle')}%(T: {self.test_class_acc['wrinkle'].sum} / F: {self.test_class_acc['wrinkle'].count - self.test_class_acc['wrinkle'].sum}) // sagging: {self.acc_avg('sagging')}%(T: {self.test_class_acc['sagging'].sum} / F: {self.test_class_acc['sagging'].count - self.test_class_acc['sagging'].sum}) // pore: {self.acc_avg('pore')}%(T: {self.test_class_acc['pore'].sum} / F: {self.test_class_acc['pore'].count - self.test_class_acc['pore'].sum}) // dryness: {self.acc_avg('dryness')}%(T: {self.test_class_acc['dryness'].sum} / F: {self.test_class_acc['dryness'].count - self.test_class_acc['dryness'].sum})"
            )

            self.logger.info(
                f"[{iter} / {len(self.test_loader)}]Total Average Acc => {((self.acc_avg('pigmentation') + self.acc_avg('wrinkle') + self.acc_avg('sagging') + self.acc_avg('pore') + self.acc_avg('dryness') ) / 5):.2f}%"
            )

        else:
            self.logger.info(
                f"count: {self.loss_avg('count')} // moisture: {self.loss_avg('moisture')} // wrinkle: {self.loss_avg('wrinkle')} // elasticity: {self.loss_avg('elasticity')} // pore: {self.loss_avg('pore')}"
            )
            self.logger.info(
                f"[{iter} / {len(self.test_loader)}] Total Average MAE => {((self.loss_avg('count') + self.loss_avg('moisture') + self.loss_avg('wrinkle') +self.loss_avg('elasticity') + self.loss_avg('pore')) / 5):.3f}"
            )

        self.logger.info("============" * 15)

    def match_img(self, vis_img, img):
        col = self.num % self.col
        row = self.num // self.col
        vis_img[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = img

        return vis_img

    def nan_detect(self, label):
        nan_list = list()
        for batch_idx, batch_data in enumerate(label):
            for value in batch_data:
                if not torch.isfinite(value):
                    nan_list.append(batch_idx)
        return nan_list

    def get_test_loss(self, pred_p, label, area_num, patch_list):
        patch_list[area_num].append(dict())
        for idx, name in enumerate(self.equip_loss[area_num]):
            dig = name.split("_")[-1]
            gt = label[:, idx: idx + 1]
            if torch.isnan(gt): 
                continue
            
            pred = pred_p[:, idx: idx + 1]
            self.test_regresion_mae[name].update(
                self.criterion(pred, gt).item(), batch_size=pred.shape[0]
            )
            self.logger.info(
                patch_list[area_num][2][0]
                + f"({dig})"
                + f"==> Pred: {pred.item():.3f}  /  Gt: {gt.item():.3f}  ==> MAE: {self.criterion(pred, gt).item():.3f}"
            )
            
            self.count[f"{area_num}_{dig}"] += 1

            if dig == "moisture":
                gt, pred = gt * 100, pred * 100
            elif dig == "count":
                gt, pred = gt * 350, pred * 350
            elif dig == "pore":
                gt, pred = gt * 3000, pred * 3000
            patch_list[area_num][3][dig] = [round(gt.item(), 3), round(pred.item(), 3)]

        return patch_list

    def get_test_acc(self, pred, label, patch_list, area_num):
        gt = (
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
            .permute(1, 0)
            .to(device)
        )
        num = 0
        patch_list[area_num].append(dict())
        for idx, area_name in enumerate(label):
            dig = area_name.split("_")[-1]
            if area_name == "forehead_wrinkle":
                pred_l = torch.argmax(pred[:, num : num + class_num_list["forehead_wrinkle"]], dim=1)
            else:
                pred_l = torch.argmax(pred[:, num : num + class_num_list[dig]], dim=1)
            num += class_num_list[dig]

            score = 0
            flag = False
            if abs((pred_l - gt[:, idx]).item()) < 2:
                score += 1
                flag = True
            self.test_class_acc[dig].update_acc(
                score,
                pred_l.shape[0],
            )
            patch_list[area_num][3][dig] = [int(gt[:, idx].item()), int(pred_l.item())]
            self.logger.info(
                patch_list[area_num][2][0]
                + f"({dig})"
                + f"==> Pred: {pred_l.item()}  /  Gt: {gt[:, idx].item()}  ==> {flag} "
            )
            
            self.count[f"{area_num}_{dig}"] += 1


        return patch_list

    def test(self, model_num_class, data_loader):
        for iter, patch_list in enumerate(data_loader):
            for model_idx in range(len(model_num_class)):
                if np.isnan(model_num_class[model_idx]):
                    continue

                self.choice(model_idx)
                self.model = self.model_list[self.m_idx]
                self.model.eval()

                data_loader = self.test_loader
                area_num = str(self.m_idx + 1) if self.flag else str(self.m_idx)

                if type(patch_list[area_num][1]) == torch.Tensor:
                    label = patch_list[area_num][1].to(device)
                else:
                    for name in patch_list[area_num][1]:
                        patch_list[area_num][1][name] = patch_list[area_num][1][
                            name
                        ].to(device)
                    label = patch_list[area_num][1]

                if label == {}:
                    continue        ## 눈가/볼 영역이 없는 경우
                
                img = patch_list[area_num][0].to(device)

                if area_num in [4, 6]:
                    img = torch.flip(img, dims=[3])

                if img.shape[-1] > 128:
                    img_l = img[:, :, :, :128]
                    img_r = img[:, :, :, 128:]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                elif img.shape[-2] > 128:
                    img_l = img[:, :, :128, :]
                    img_r = img[:, :, 128:, :]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                else:
                    pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    _ = self.get_test_acc(pred, label, patch_list, area_num)

                else:
                    _ = self.get_test_loss(pred, label.to(device), area_num, patch_list)
            self.print_total(iter)
            
        if self.args.log: [self.logger.info(f"{key} => {self.count[key]} 장") for key in self.count]
