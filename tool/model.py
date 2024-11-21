from collections import defaultdict
from copy import deepcopy
from datetime import datetime
import random
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr

from tqdm import tqdm
from data_loader import mkdir
import torch.optim as optim
from utils import (
    AverageMeter,
    FocalLoss,
    mape_loss,
    save_checkpoint,
    save_image,
    CB_loss
)
import os
import torch.autograd as autograd

from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error

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
        grade_num,
        info,
    ):
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
            self.grade_num,
            self.info
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
            grade_num,
            info,
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
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=20
        )

    def acc_avg(self, name):
        return round(self.test_value[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_value[name].avg, 4)
    
    def print_best(self):
        self.logger.info(
                    f"Best Epoch: {self.best_epoch}  Acc: {(self.acc_ * 100):.2f}%  Correlation: {self.corre_:.2f}"
                    )
        
        for grade in sorted(self.correct_):
            print(f"[Grade {grade}]  {self.correct_[grade]} / {self.all_[grade]} => {(self.correct_[grade] / self.all_[grade] * 100):.2f}%      ", end = "")
        print("")

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

            f_pred = list()
            f_gt = list()
            
            correct_ = defaultdict(int)
            all_ = defaultdict(int)
            (pred_, gt_) = (self.pred, self.gt) if self.phase == "Valid" else (self.pred_t, self.gt_t)
            
            for value, value2 in zip(pred_, gt_):
                for v, v1 in zip(value, value2):
                    f_pred.append(v)
                    f_gt.append(v1)

            correlation, _ = pearsonr(f_gt, f_pred)

            if self.args.mode == "class":
                (
                    micro_precision,
                    _,
                    micro_f1,
                    _,
                ) = precision_recall_fscore_support(
                    f_gt, f_pred, average="micro", zero_division=1
                )
                
                for idx, i in enumerate(f_gt):
                    all_[i] += 1
                    if i == f_pred[idx]:
                        correct_[i] += 1

                info_m = f"[Lr: {self.optimizer.param_groups[0]['lr']:4f}][Gamma: {self.args.gamma}][Early Stop: {self.update_c}/{self.args.stop_early}]" if self.phase == "Train" else ""
                self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}]{info_m}[{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}, Correlation: {correlation:.2f} micro Precision: {(micro_precision * 100):.2f}%"
                )          

                grade_ = sorted(list(all_.keys()))
                for grade in grade_:
                    print(f"        [Grade {grade}]  {correct_[grade]} / {all_[grade]} => {(correct_[grade] / all_[grade] * 100):.2f}%"  , end = "")
                print("")
                
                if self.phase == "Valid":
                    if self.best_loss[self.m_dig] > self.val_loss.avg:
                        self.best_loss[self.m_dig] = round(self.val_loss.avg, 4)
                        save_checkpoint(self,  correct_, all_, micro_precision, correlation)
                    else:
                        self.update_c += 1
                    
            else:
                self.logger.info(
                f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}][Lr: {self.optimizer.param_groups[0]['lr']:4f}][Early Stop: {self.update_c}/{self.args.stop_early}][{self.m_dig}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}, Correlation: {correlation:.2f}"
                )


    def stop_early(self):
        if self.update_c > self.args.stop_early:
            mkdir(
                os.path.join(
                    self.args.root_path, 
                    "checkpoint",
                    self.args.git_name,
                    self.args.mode,
                    self.args.name,
                    "save_model",
                    str(self.m_dig),
                    "done",
                )
            )
            return True

    def class_loss(self, pred, gt):
        loss = self.criterion(pred, gt)

        with torch.no_grad():
            pred_v = [item.argmax().item() for item in pred]
            gt_v = [item.item() for item in gt]
            
            if self.phase == "Train":
                self.pred_t.append(pred_v)
                self.gt_t.append(gt_v)
                self.train_loss.update(loss.item(), batch_size=pred.shape[0])

            elif self.phase == "Valid":
                self.pred.append(pred_v)
                self.gt.append(gt_v)
                self.val_loss.update(loss.item(), batch_size=pred.shape[0])

        return loss

    def regression(self, pred, gt):
        pred = pred.flatten()
        loss = self.criterion(pred, gt)

        with torch.no_grad():
            pred_v = [item.item() for item in pred]
            gt_v = [item.item() for item in gt]

            if self.phase == "Train":
                self.pred_t.append(pred_v)
                self.gt_t.append(gt_v)
                self.train_loss.update(loss.item(), batch_size=pred.shape[0])

            elif self.phase == "Valid":
                self.pred.append(pred_v)
                self.gt.append(gt_v)
                self.val_loss.update(loss.item(), batch_size=pred.shape[0])

        return loss

    def up_and_down(self, name, color="\033[95m", c_color="\033[0m"):
        if self.args.mode == "class":
            sub = (self.test_value[name].avg * 100) - self.keep_acc[name]
            value = round(sub, 2)
            result = (
                f"{color}+{value}{c_color}%"
                if value > 0
                else "No change" if value == 0 else f"{color}{value}{c_color}%"
            )
        else:
            sub = (self.test_value[name].avg) - self.keep_mae[name]
            value = round(sub, 4)
            result = (
                f"{color}+{value}{c_color}"
                if value > 0
                else "No change" if value == 0 else f"{color}{value}{c_color}"
            )

        return result

    def reset_log(self, flag):
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        if flag: self.epoch += 1
        self.pred = list()
        self.gt = list()
        self.pred_t = list()
        self.gt_t = list()

    def update_e(self, epoch, correct_, all_, micro_precision, correlation):
        self.epoch = epoch
        self.correct_ = correct_
        self.all_ = all_
        self.acc_ = micro_precision
        self.corre_ = correlation
        
    def train(self):
        self.model.train()
        self.phase = "Train"
        self.criterion = (
            CB_loss(samples_per_cls=self.grade_num, no_of_classes=len(self.grade_num), gamma = self.args.gamma)
            if self.args.mode == "class"
            else nn.L1Loss()
        )
        self.prev_model = deepcopy(self.model)
        random_num = random.randrange(0, len(self.train_loader))

        for self.iter, (img, label, self.img_names, _, _, _) in enumerate(
            self.train_loader
        ): 
            img, label = img.to(device), label.to(device)

            pred = self.model(img)
            
            if self.args.mode == "class":
                loss = self.class_loss(pred, label)
            else:
                loss = self.regression(pred, label)
                
            if self.iter == random_num:
                save_image(self, img)

            self.print_loss(len(self.train_loader))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if torch.isnan(pred).any() or torch.isnan(loss).any():
                self.optimizer.param_groups[0]["lr"] /= 2
                self.model = deepcopy(self.prev_model)
                self.optimizer.param_groups[0]['params'] = self.model.parameters()
                
                return True
            
        self.print_loss(len(self.train_loader), final_flag=True)
        return False

    def valid(self):
        self.phase = "Valid"
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )
        random_num = random.randrange(0, len(self.valid_loader))
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, self.img_names, _, meta_v, _) in enumerate(
                self.valid_loader
            ):
                img, label = img.to(device), label.to(device)
                pred = self.model(img)
                
                if self.args.mode == "class":
                    self.class_loss(pred, label)
                else:
                    self.regression(pred, label)
                    
                if self.iter == random_num:
                    save_image(self, img)
                    
                self.print_loss(len(self.valid_loader))

            self.scheduler.step(self.val_loss.avg)
            self.print_loss(len(self.valid_loader), final_flag=True)


class Model_test(Model):
    def __init__(self, args, logger):
        self.args = args
        self.pred = defaultdict(lambda: defaultdict(list))
        self.gt = defaultdict(lambda: defaultdict(list))
        self.logger = logger

    def test(self, model, testset_loader, key):
        self.model = model
        self.testset_loader = testset_loader
        self.m_dig = key
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, self.img_names, self.digs, _, _) in enumerate(
                tqdm(self.testset_loader, desc=self.m_dig)
            ):
                img, label = img.to(device), label.to(device)

                pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    self.get_test_acc(pred, label)
                else:
                    self.get_test_loss(pred, label)

    def save_value(self):
        pred_path = os.path.join(self.args.root_path , "checkpoint", self.args.git_name, self.args.mode, self.args.name, "prediction")
        mkdir(pred_path)
        with open(os.path.join(pred_path, f"pred.txt"), "w") as p:
            with open(os.path.join(pred_path, f"gt.txt"), "w") as g:
                for key in list(self.pred.keys()):
                    for angle in self.pred[key].keys():
                        for p_v, g_v in zip(self.pred[key][angle], self.gt[key][angle]):
                            p.write(f"{angle}, {key}, {p_v[0]}, {p_v[1]} \n")
                            g.write(f"{angle}, {key}, {g_v[0]}, {g_v[1]} \n")
        g.close()
        p.close()

    def print_test(self):
        for angle in self.pred[self.m_dig].keys():
            gt_v = [value[0] for value in self.gt[self.m_dig][angle]]
            pred_v = [value[0] for value in self.pred[self.m_dig][angle]]
            
            correct_ = defaultdict(int)
            all_ = defaultdict(int)
            
            for gt, pred in zip(gt_v, pred_v):
                all_[gt] += 1
                if gt == pred:
                    correct_[gt] += 1
            
            correlation, p_value = pearsonr(gt_v, pred_v)
            
            if self.args.mode == "regression":
                n_gt_v = [value[0]/value[-1] for value in self.gt[self.m_dig]]
                n_pred_v = [value[0]/value[-1] for value in self.pred[self.m_dig]]
                
                mae = mean_absolute_error(gt_v, pred_v)
                mape = mape_loss()(np.array(pred_v), np.array(gt_v))
                nmae = mean_absolute_error(n_gt_v, n_pred_v)
                self.logger.info(
                    f"[{self.m_dig}]Correlation: {correlation:.2f}, P-value: {p_value:.4f}, MAE: {mae:.4f}, MAPE: {mape:.3f}, NMAE: {nmae:.3f}\n"
                )
            
            else:
                mae_ = [abs(p-g) for p, g in zip(pred_v, gt_v)]
                mae_ = sum(mae_) / len(mae_)
                
                mae_0 = [True if abs(p-g) ==0 else False for p, g in zip(pred_v, gt_v)]
                mae_0 = sum(mae_0) / len(mae_0)
                
                mae_1 = [True if abs(p-g) <= 1 else False for p, g in zip(pred_v, gt_v)]
                mae_1 = sum(mae_1) / len(mae_1)
                
                mae_2 = [True if abs(p-g) <= 2 else False for p, g in zip(pred_v, gt_v)]
                mae_2 = sum(mae_2) / len(mae_2)

                self.logger.info(
                    f"[{angle}][{self.m_dig}]Correlation: {correlation:.2f}, P-value: {p_value:.4f}, MAE: {mae_:.2f}, MAE(==0): {mae_0 * 100:.2f}%,  MAE(=<1): {mae_1 * 100:.2f}%, MAE(=<2): {mae_2 * 100:.2f}%"
                )
                
                for grade in all_:
                    self.logger.info(
                        f"          {grade} grade Acc: {correct_[grade]} / {all_[grade]} -> {(correct_[grade]/all_[grade] * 100):.2f} %"
                    )
                
                with open(f"{self.args.log_path}/print_{angle}.txt", "a") as f:
                    if self.m_dig == "dryness": f.write(f"Angle, Area, Correlation, P-value, MAE, MAE(==0), MAE(=<1), MAE(=<2)\n")                
                    f.write(f"{angle}, {self.m_dig}, {correlation:.2f}, {p_value:.4f}, {mae_:.2f}, {mae_0 * 100:.2f}, {mae_1 * 100:.2f}, {mae_2 * 100:.2f}\n")                

    def get_test_loss(self, pred, gt):
        if "elasticity_R2" in self.m_dig:
            value = 1

        elif "moisture" in self.m_dig:
            value = 100

        elif "wrinkle_Ra" in self.m_dig:
            value = 50

        elif self.m_dig == "pigmentation":
            value = 350

        elif "pore" in self.m_dig:
            value = 2600

        else:
            assert 0, "error"

        for idx, (pred_item, gt_item) in enumerate(zip(pred, gt)):
            self.pred[self.m_dig].append(
                [round(pred_item.item() * value, 3), self.img_names[idx], value]
            )
            self.gt[self.m_dig].append(
                [round(gt_item.item() * value, 3), self.img_names[idx], value]
            )

    def get_test_acc(self, pred, gt):
        for idx, (pred_item, gt_item) in enumerate(zip(pred, gt)):
            self.pred[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [pred_item.argmax().item(), self.img_names[idx]]
            )
            self.gt[self.m_dig][self.img_names[idx].split("_")[-3]].append([gt_item.item(), self.img_names[idx]])
            
            
            
            
