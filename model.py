from collections import defaultdict
import inspect
import random
import torch
import torch.nn as nn
import numpy as np
import copy

from tqdm import tqdm
from data_loader import class_num_list, mkdir
import torch.optim as optim
from utils import (
    AverageMeter,
    FocalLoss,
    save_checkpoint,
    save_image,
)
import os
from torch.utils import data
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Model(object):
    def __init__(
        self,
        args,
        model,
        train_loader,
        valid_loader,
        logger,
        check_path,
        model_num_class,
        writer,
        dig_k,
    ):
        super().__init__()
        (
            self.args,
            self.model,
            self.temp_model,
            self.train_loader,
            self.valid_loader,
            self.best_loss,
            self.logger,
            self.check_path,
            self.model_num_class,
            self.writer,
            self.m_dig,
        ) = (
            args,
            model,
            None,
            train_loader,
            valid_loader,
            args.best_loss,
            logger,
            check_path,
            model_num_class,
            writer,
            dig_k,
        )

        self.train_loss, self.val_loss = AverageMeter(), AverageMeter()
        self.keep_acc = {
            "sagging": AverageMeter(),
            "wrinkle_forehead": AverageMeter(),
            "wrinkle_glabellus": AverageMeter(),
            "wrinkle_perocular": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation_forehead": AverageMeter(),
            "pigmentation_cheek": AverageMeter(),
            "dryness": AverageMeter(),
        }
        self.keep_mae = {
            "moisture": 0,
            "wrinkle": 0,
            "elasticity": 0,
            "pore": 0,
            "count": 0,
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
        self.epoch = 0
        self.criterion = (
            FocalLoss(gamma=self.args.gamma)
            if self.args.mode == "class"
            else nn.L1Loss()
        )

        (
            self.phase,
            self.update_c,
            self.stop_loss,
            self.device,
        ) = (None, 0, np.inf, device)
        self.pred, self.gt = list(), list()
        self.pred_t, self.gt_t = list(), list()

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min")

    def acc_avg(self, name):
        return round(self.test_value[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_value[name].avg, 4)

    def print_loss(self, dataloader_len, final_flag=False):
        print(
            f"\rEpoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}",
            end="",
        )
        if final_flag:
            self.writer.add_scalar(
                f"{self.phase}/{self.m_dig}",
                self.train_loss.avg if self.phase == "Train" else self.val_loss.avg,
                self.epoch,
            )

            if self.phase == "Valid":
                if self.best_loss[self.m_dig] > self.val_loss.avg:
                    self.best_loss[self.m_dig] = round(self.val_loss.avg, 4)
                    save_checkpoint(
                        self.model, self.args, self.epoch, self.m_dig, self.best_loss
                    )
                    self.update_c = 0
                else:
                    self.update_c += 1

                f_pred = list()
                f_gt = list()
                for (key, value), (_, value2) in zip(self.pred, self.gt):
                    if key != self.m_dig:
                        continue
                    for v, v1 in zip(value, value2):
                        f_pred.append(v)
                        f_gt.append(v1)

                (
                    macro_precision,
                    macro_recall,
                    macro_f1,
                    _,
                ) = precision_recall_fscore_support(
                    f_gt, f_pred, average="macro", zero_division=1
                )
                acc = accuracy_score(f_gt, f_pred)
                self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.val_loss.avg:.04f}"
                )
                self.logger.info(
                    f"[{self.m_dig}][Lr: {self.optimizer.param_groups[0]['lr']:4f}] [Early Stop: {self.update_c}/{self.args.stop_early}] Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, Acc: {acc * 100:.2f}%"
                )
            else:
                self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg:.04f}"
                )

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            return True

    def class_loss(self, pred, gt):
        loss = self.criterion(pred, gt)

        pred_v = [item.argmax().item() for item in pred]
        gt_v = [item.item() for item in gt]

        if self.phase == "Train":
            self.pred_t.append([self.m_dig, pred_v])
            self.gt_t.append([self.m_dig, gt_v])
            self.train_loss.update(loss.item(), batch_size=pred.shape[0])

        elif self.phase == "Valid":
            self.pred.append([self.m_dig, pred_v])
            self.gt.append([self.m_dig, gt_v])
            self.val_loss.update(loss.item(), batch_size=pred.shape[0])
            
        return loss

    def regression(self, pred, label, dig):
        loss = self.criterion(pred[0], label.to(device))

        self.pred.append([dig, pred[0].item()])
        self.gt.append([dig, label.item()])

        self.train_loss[self.m_dig].update(
            loss.item(), batch_size=pred.shape[0]
        ) if self.phase == "Train" else self.val_loss[self.m_dig].update(
            loss.item(), batch_size=pred.shape[0]
        )

        return loss

    def up_and_down(self, name, color="\033[95m", c_color="\033[0m"):
        if self.args.mode == "class":
            sub = (self.test_value[name].avg * 100) - self.keep_acc[name]
            value = round(sub, 2)
            result = (
                f"{color}+{value}{c_color}%"
                if value > 0
                else "No change"
                if value == 0
                else f"{color}{value}{c_color}%"
            )
        else:
            sub = (self.test_value[name].avg) - self.keep_mae[name]
            value = round(sub, 4)
            result = (
                f"{color}+{value}{c_color}"
                if value > 0
                else "No change"
                if value == 0
                else f"{color}{value}{c_color}"
            )

        return result

    def save_value(self):
        pred_path = os.path.join(self.check_path, "prediction")
        mkdir(pred_path)
        with open(
            os.path.join(pred_path, f"epoch_{self.epoch}_pred_train.txt"), "w"
        ) as p:
            with open(
                os.path.join(pred_path, f"epoch_{self.epoch}_gt_train.txt"), "w"
            ) as g:
                for idx in range(len(self.pred_t)):
                    p.write(f"{self.pred_t[idx][0]}, {self.pred_t[idx][1]} \n")
                    g.write(f"{self.gt_t[idx][0]}, {self.gt_t[idx][1]} \n")
        g.close()
        p.close()

        with open(
            os.path.join(pred_path, f"epoch_{self.epoch}_pred_val.txt"), "w"
        ) as p:
            with open(
                os.path.join(pred_path, f"epoch_{self.epoch}_gt_val.txt"), "w"
            ) as g:
                for idx in range(len(self.pred)):
                    p.write(f"{self.pred[idx][0]}, {self.pred[idx][1]} \n")
                    g.write(f"{self.gt[idx][0]}, {self.gt[idx][1]} \n")
        g.close()
        p.close()

    def reset_log(self):
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.pred = list()
        self.gt = list()

    def update_e(self, epoch):
        self.epoch = epoch

    def train(self):
        self.model.train()
        self.phase = "Train"
        random_num = random.randrange(0, len(self.train_loader))

        for self.iter, (img, label, self.img_names, _, meta_v) in enumerate(
            self.train_loader
        ):
            img, label = img.to(device), label.to(device)
            pred = self.model(img, meta_v)

            if self.args.mode == "class":
                loss = self.class_loss(pred, label)
            else:
                loss = self.regression(pred, label)

            if self.args.img:
                save_image(self, img)
            else:
                if self.iter == random_num:
                    save_image(self, img)

            self.print_loss(len(self.train_loader))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        self.print_loss(len(self.train_loader), final_flag=True)

    def valid(self):
        self.phase = "Valid"
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, self.img_names, _, meta_v) in enumerate(
                self.valid_loader
            ):
                img, label = img.to(device), label.to(device)
                pred = self.model(img, meta_v)

                if self.args.mode == "class":
                    self.class_loss(pred, label)
                else:
                    self.regression(pred, label)

                self.print_loss(len(self.valid_loader))

            self.scheduler.step(self.val_loss.avg)
            self.print_loss(len(self.valid_loader), final_flag=True)


class Model_test(Model):
    def __init__(self, args, logger):
        self.args = args
        self.pred = defaultdict(list)
        self.gt = defaultdict(list)
        self.logger = logger

    def test(self, model, testset_loader, key):
        self.model = model
        self.testset_loader = testset_loader
        self.m_dig = key
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, _, _, meta_v) in enumerate(
                tqdm(self.testset_loader, desc=self.m_dig)
            ):
                img, label = img.to(device), label.to(device)
                pred = self.model.to(device)(img, meta_v)

                if self.args.mode == "class":
                    self.get_test_acc(pred, label)
                else:
                    self.get_test_loss(pred, label)

    def save_value(self):
        pred_path = os.path.join(self.agrs.check_path, "prediction")
        mkdir(pred_path)
        with open(os.path.join(pred_path, f"pred.txt"), "w") as p:
            with open(os.path.join(pred_path, f"gt.txt"), "w") as g:
                for key in list(self.pred.keys()):
                    for p_v, g_v in zip(self.pred[key], self.gt[key]):
                        p.write(f"{key}, {p_v} \n")
                        g.write(f"{key}, {g_v} \n")
        g.close()
        p.close()

    def print_test(self):
        (
            macro_precision,
            macro_recall,
            macro_f1,
            _,
        ) = precision_recall_fscore_support(
            self.pred[self.m_dig], self.gt[self.m_dig], average="macro", zero_division=1
        )
        acc = accuracy_score(self.gt[self.m_dig], self.pred[self.m_dig])
        self.logger.info(
            f"[{self.m_dig}] Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, Acc: {acc * 100:.2f}% "
        )

    def get_test_loss(self, pred, gt):
        self.test_regresion_mae[self.m_dig].update(
            self.criterion(pred[0], gt).item(), batch_size=pred.shape[0]
        )

        self.pred.append([self.m_dig, pred.item()])
        self.gt.append([self.m_dig, gt.item()])

    def get_test_acc(self, pred, gt):
        [self.pred[self.m_dig].append(item.argmax().item()) for item in pred]
        [self.gt[self.m_dig].append(item.item()) for item in gt]
