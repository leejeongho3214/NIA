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


def save_checkpoint(model, args, epoch, m_dig, best_loss):
    checkpoint_dir = os.path.join(args.output_dir, args.mode, args.name, str(m_dig))
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

        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_dig = 0
        self.model = None
        self.update_c = 0
        
        self.pred = list()
        self.gt = list()

    def choice(self, m_dig):
        self.model = copy.deepcopy(self.model_list[m_dig])
        self.m_dig = m_dig

    def acc_avg(self, name):
        return round(self.test_class_acc[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_regresion_mae[name].avg, 4)

    def print_total(self):
        if self.args.mode == "class":
            self.logger.info(
                f"pigmentation: {self.acc_avg('pigmentation')}%(T: {self.test_class_acc['pigmentation'].sum} / F: {self.test_class_acc['pigmentation'].count - self.test_class_acc['pigmentation'].sum}) // wrinkle: {self.acc_avg('wrinkle')}%(T: {self.test_class_acc['wrinkle'].sum} / F: {self.test_class_acc['wrinkle'].count - self.test_class_acc['wrinkle'].sum}) // sagging: {self.acc_avg('sagging')}%(T: {self.test_class_acc['sagging'].sum} / F: {self.test_class_acc['sagging'].count - self.test_class_acc['sagging'].sum}) // pore: {self.acc_avg('pore')}%(T: {self.test_class_acc['pore'].sum} / F: {self.test_class_acc['pore'].count - self.test_class_acc['pore'].sum}) // dryness: {self.acc_avg('dryness')}%(T: {self.test_class_acc['dryness'].sum} / F: {self.test_class_acc['dryness'].count - self.test_class_acc['dryness'].sum})"
            )
            self.logger.info(
                f"Total Average Acc => {((self.acc_avg('pigmentation') + self.acc_avg('wrinkle') + self.acc_avg('sagging') + self.acc_avg('pore') + self.acc_avg('dryness') ) / 5):.2f}%"
            )
        else:
            self.logger.info(
                f"count: {self.loss_avg('count')} // moisture: {self.loss_avg('moisture')} // wrinkle: {self.loss_avg('wrinkle')} // elasticity: {self.loss_avg('elasticity')} // pore: {self.loss_avg('pore')}"
            )
            self.logger.info(
                f"Total Average MAE => {((self.loss_avg('count') + self.loss_avg('moisture') + self.loss_avg('wrinkle') +self.loss_avg('elasticity') + self.loss_avg('pore')) / 5):.3f}"
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

    def get_test_loss(self, pred, gt):
        self.test_regresion_mae[self.m_dig].update(
            self.criterion(pred, gt).item(), batch_size=pred.shape[0]
        )
        
        if self.m_dig in ["moisture", "wrinkle"]:
            gt, pred = gt * 100, pred * 100
        elif self.m_dig == "count":
            gt, pred = gt * 350, pred * 350
        elif self.m_dig == "pore":
            gt, pred = gt * 3000, pred * 3000
        
        self.pred.append([self.m_dig, pred.item()])
        self.gt.append([self.m_dig, gt.item()])

            
    def get_test_acc(self, pred, gt, patch_list):

        pred_g = pred.argmax().item()
        self.pred.append([self.m_dig, pred_g])
        self.gt.append([self.m_dig, gt.item()])
        
        if abs((pred_g - gt).item()) == 0:
            score = 1
        else:
            score = 0 
            
        self.test_class_acc[self.m_dig].update_acc(
            score,
            1,
        )

        return patch_list

    def save_value(self):
        path = os.path.join("predictino", self.args.name)
        mkdir(path)
        with open(
            os.path.join(path, f"pred.txt"), "w"
        ) as p:
            with open(
                os.path.join(path, f"gt.txt"), "w"
            ) as g:
                for idx in range(len(self.pred)):
                    p.write(f"{self.pred[idx][0]}, {self.pred[idx][1]} \n")
                    g.write(f"{self.gt[idx][0]}, {self.gt[idx][1]} \n")
        g.close()
        p.close()

    def test(self, model_num_class, data_loader):

        data_loader = self.test_loader
        len_dataset = len(data_loader) * len(model_num_class)
        total_iter = 0
        for dig in model_num_class:
            self.choice(dig)
            self.model = self.model_list[self.m_dig]
            self.model.eval()
            for patch_list in data_loader:
                for item in patch_list[self.m_dig]:
                    if type(item[1]) == torch.Tensor:
                        label = item[1].to(device)
                    else:
                        for name in item[1]:
                            item[1][name] = item[1][
                                name
                            ].to(device)
                        label = item[1]

                    if label == {}:
                        continue        ## 눈가/볼 영역이 없는 경우
                    img = item[0].to(device)

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
                        _ = self.get_test_acc(pred, label, patch_list)
                    else:
                        _ = self.get_test_loss(pred, label.to(device))
                        
                total_iter += 1 
                print(f"count =>->=> {total_iter}/{len_dataset}", end="\r")
                
        self.print_total()
        if self.args.log: [self.logger.info(f"{key} => {self.count[key]} 장") for key in self.count]
