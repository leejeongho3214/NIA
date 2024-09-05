from collections import defaultdict
import random
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
            self.grade_num
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
                    save_checkpoint(self)
                    self.update_c = 0
                else:
                    self.update_c += 1

                f_pred = list()
                f_gt = list()
                for value, value2 in zip(self.pred, self.gt):
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

                    self.logger.info(
                        f"Epoch: {self.epoch} [{self.phase}][Lr: {self.optimizer.param_groups[0]['lr']:4f}][Gamma: {self.args.gamma}][Early Stop: {self.update_c}/{self.args.stop_early}][{self.m_dig}] micro Precision: {(micro_precision * 100):.2f}%, micro F1: {micro_f1:.4f}"
                    )
                    self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.val_loss.avg:.04f}, Correlation: {correlation:.2f}"
                    )
                else:
                    self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}][Lr: {self.optimizer.param_groups[0]['lr']:4f}][Early Stop: {self.update_c}/{self.args.stop_early}][{self.m_dig}] ---- >  loss: {self.val_loss.avg:.04f}, Correlation: {correlation:.2f}"
                    )

            else:
                self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}"
                )

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            mkdir(
                os.path.join(
                    self.args.output_dir,
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
        self.criterion = (
            CB_loss(samples_per_cls=self.grade_num, no_of_classes=len(self.grade_num), gamma = self.args.gamma)
            if self.args.mode == "class"
            else nn.L1Loss()
        )
        random_num = random.randrange(0, len(self.train_loader))

        for self.iter, (img, label, self.img_names, _, meta_v, _) in enumerate(
            self.train_loader
        ):
            img, label = img.to(device), label.to(device)

            pred = self.model(img, meta_v)
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

        self.print_loss(len(self.train_loader), final_flag=True)

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
                pred = self.model(img, meta_v)

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
        self.pred = defaultdict(list)
        self.gt = defaultdict(list)
        self.logger = logger

    def test(self, model, testset_loader, key):
        self.model = model
        self.testset_loader = testset_loader
        self.m_dig = key
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, self.img_names, self.digs, meta_v, _) in enumerate(
                tqdm(self.testset_loader, desc=self.m_dig)
            ):
                img, label = img.to(device), label.to(device)

                pred = (
                    self.model.to(device)(img, meta_v)
                    if self.args.model != "coatnet"
                    else self.model.to(device)(img)
                )

                if self.args.mode == "class":
                    self.get_test_acc(pred, label)
                else:
                    self.get_test_loss(pred, label)

    def save_value(self):
        pred_path = os.path.join(self.args.check_path, "prediction")
        mkdir(pred_path)
        with open(os.path.join(pred_path, f"pred.txt"), "w") as p:
            with open(os.path.join(pred_path, f"gt.txt"), "w") as g:
                for key in list(self.pred.keys()):
                    for p_v, g_v in zip(self.pred[key], self.gt[key]):
                        p.write(f"{key}, {p_v[0]}, {p_v[1]} \n")
                        g.write(f"{key}, {g_v[0]}, {g_v[1]} \n")
        g.close()
        p.close()

    def print_test(self):
        gt_v = [value[0] for value in self.gt[self.m_dig]]
        pred_v = [value[0] for value in self.pred[self.m_dig]]
        
        if self.args.mode == "regression":
            n_gt_v = [value[0]/value[-1] for value in self.gt[self.m_dig]]
            n_pred_v = [value[0]/value[-1] for value in self.pred[self.m_dig]]
        
        
        if self.args.mode == "class":
            for gt_value, pred_value in [(gt_v, pred_v)]:
                (
                    micro_precision,
                    _,
                    _,
                    _,
                ) = precision_recall_fscore_support(
                    gt_value,
                    pred_value,
                    average="micro",
                    zero_division=0
                )

                correlation, p_value = pearsonr(gt_value, pred_value)

                top_3 = [
                    False if abs(g - p) > 1 else True for g, p in zip(gt_value, pred_value)
                ]
                top_3_acc = sum(top_3) / len(top_3)

                self.logger.info(
                    f"[{self.m_dig}]Acc: {micro_precision:.4f} Correlation: {correlation:.2f}, P-value: {p_value:.4f}, Top-3 Acc: {top_3_acc:.4f}\n"
                )

        else:
            correlation, p_value = pearsonr(gt_v, pred_v)
            mae = mean_absolute_error(gt_v, pred_v)
            mape = mape_loss()(np.array(pred_v), np.array(gt_v))
            nmae = mean_absolute_error(n_gt_v, n_pred_v)
            self.logger.info(
                f"[{self.m_dig}]Correlation: {correlation:.2f}, P-value: {p_value:.4f}, MAE: {mae:.4f}, MAPE: {mape:.3f}, NMAE: {nmae:.3f}\n"
            )


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
            self.pred[self.m_dig].append(
                [pred_item.argmax().item(), self.img_names[idx]]
            )
            self.gt[self.m_dig].append([gt_item.item(), self.img_names[idx]])
