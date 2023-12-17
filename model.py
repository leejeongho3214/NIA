from collections import defaultdict
import random
import torch
import torch.nn as nn
import numpy as np
import copy
from data_loader import class_num_list
from utils import (
    adjust_learning_rate,
    get_item,
    AverageMeter,
    pred_image,
    save_checkpoint,
    LabelSmoothingCrossEntropy,
    save_image,
    softmax,
)
import os
from utils import labeling
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Model(object):
    def __init__(
        self,
        args,
        model_list,
        train_loader,
        valid_loader,
        logger,
        check_path,
    ):
        super(Model, self).__init__()
        (
            self.args,
            self.model_list,
            self.temp_model_list,
            self.train_loader,
            self.valid_loader,
            self.best_loss,
            self.logger,
            self.check_path,
        ) = (
            args,
            model_list,
            defaultdict(str),
            train_loader,
            valid_loader,
            args.best_loss,
            logger,
            check_path,
        )

        self.train_loss = (
            {
                "sagging": AverageMeter(),
                "wrinkle": AverageMeter(),
                "pore": AverageMeter(),
                "pigmentation": AverageMeter(),
                "dryness": AverageMeter(),
            }
            if self.args.mode == "class"
            else {
                "moisture": AverageMeter(),
                "wrinkle": AverageMeter(),
                "elasticity": AverageMeter(),
                "pore": AverageMeter(),
                "count": AverageMeter(),
            }
        )

        self.val_loss = copy.deepcopy(self.train_loss)
        self.class_acc = {
            "sagging": AverageMeter(),
            "wrinkle": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation": AverageMeter(),
            "dryness": AverageMeter(),
        }

        self.keep_acc = {
            "sagging": 0,
            "wrinkle": 0,
            "pore": 0,
            "pigmentation": 0,
            "dryness": 0,
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
            (
                LabelSmoothingCrossEntropy(self.args.smooth)
                if not self.args.cross
                else nn.CrossEntropyLoss()
            )
            if self.args.mode == "class"
            else nn.L1Loss()
        )
        (
            self.phase,
            self.m_dig,
            self.model,
            self.update_c,
            self.stop_loss,
            self.device,
        ) = (None, 0, None, 0, np.inf, device)
        self.pred, self.gt = list(), list()
        self.pred_t, self.gt_t = list(), list()

    def choice(self, m_dig):
        self.model = copy.deepcopy(self.model_list[m_dig])
        self.m_dig = m_dig

    def acc_avg(self, name):
        return round(self.test_value[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_value[name].avg, 4)

    def print_loss(self, iteration, num_dig, final_flag=False):
        dataloader_len = (
            len(self.train_loader) * num_dig
            if self.phase == "train"
            else len(self.valid_loader) * num_dig
        )
        print(
            f"\rEpoch: {self.epoch} [{self.phase}][{self.m_dig}][{iteration}/{dataloader_len}] ---- >  loss: {self.train_loss[self.m_dig].avg if self.phase == 'train' else self.val_loss[self.m_dig].avg:.04f}",
            end="",
        )

        if final_flag:
            self.logger.info(
                f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{iteration}/{dataloader_len}] ---- >  loss: {self.train_loss[self.m_dig].avg if self.phase == 'train' else self.val_loss[self.m_dig].avg:.04f}"
            )

            self.temp_model_list[self.m_dig] = self.model
            if self.phase == "valid":
                f_pred = defaultdict(list)
                f_gt = defaultdict(list)
                for (key, value), (_, value2) in zip(self.pred, self.gt):
                    f_pred[key].append(value)
                    f_gt[key].append(value2)
                (
                    macro_precision,
                    macro_recall,
                    macro_f1,
                    _,
                ) = precision_recall_fscore_support(
                    f_gt[key], f_pred[key], average="macro", zero_division=1
                )
                acc = accuracy_score(f_gt[key], f_pred[key])
                self.logger.info(
                    f"[{key}] Macro Precision: {macro_precision:.4f}, Macro Recall: {macro_recall:.4f}, Macro F1: {macro_f1:.4f}, Acc: {acc * 100:.2f}% "
                )

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            return True
        
    def class_loss(self, pred, label, dig, num_class):
        if self.args.cross:
            gt = label
            loss = self.criterion(pred, gt) 
        else:
            gt = labeling(label, num_class, dig).cuda()
            loss = self.criterion(pred, gt, dig) 
        pred_p = softmax(pred)
        
        if abs((pred_p.argmax().item() - gt.item())) == 0:
            score = 1
        else:
            score = 0

        if self.phase == "valid":
            self.pred.append([dig, pred_p.argmax().item()])
            self.gt.append([dig, gt.item()])
            self.class_acc[self.m_dig].update_acc(score, 1)

        elif self.phase == "train":
            self.pred_t.append([dig, pred_p.argmax().item()])
            self.gt_t.append([dig, gt.item()])

        self.train_loss[self.m_dig].update(
            loss.item(), batch_size=1
        ) if self.phase == "train" else self.val_loss[self.m_dig].update(
            loss.item(), batch_size=1
        )


        return loss

    def regression(self, pred, label, dig):
        loss = self.criterion(pred[0], label.to(device))

        self.pred.append([dig, pred[0].item()])
        self.gt.append([dig, label.item()])

        self.train_loss[self.m_dig].update(
            loss.item(), batch_size=pred.shape[0]
        ) if self.phase == "train" else self.val_loss[self.m_dig].update(
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
        with open(
            os.path.join(self.check_path, f"epoch_{self.epoch}_pred_train.txt"), "w"
        ) as p:
            with open(
                os.path.join(self.check_path, f"epoch_{self.epoch}_gt_train.txt"), "w"
            ) as g:
                for idx in range(len(self.pred_t)):
                    p.write(f"{self.pred_t[idx][0]}, {self.pred_t[idx][1]} \n")
                    g.write(f"{self.gt_t[idx][0]}, {self.gt_t[idx][1]} \n")
        g.close()
        p.close()

    def save_value1(self):
        with open(
            os.path.join(self.check_path, f"epoch_{self.epoch}_pred_val.txt"), "w"
        ) as p:
            with open(
                os.path.join(self.check_path, f"epoch_{self.epoch}_gt_val.txt"), "w"
            ) as g:
                for idx in range(len(self.pred)):
                    p.write(f"{self.pred[idx][0]}, {self.pred[idx][1]} \n")
                    g.write(f"{self.gt[idx][0]}, {self.gt[idx][1]} \n")
        g.close()
        p.close()

    def update_m(self, model_num_class):
        best_loss = 0
        for item in model_num_class:
            if self.best_loss[item] > self.val_loss[item].avg:
                try:
                    self.best_loss[item] = round(self.val_loss[item].avg.item(), 4)
                except:
                    self.best_loss[item] = round(self.val_loss[item].avg, 4)

                self.model_list[item] = copy.deepcopy(self.temp_model_list[item])
                save_checkpoint(
                    self.model_list[item], self.args, self.epoch, item, self.best_loss
                )
            best_loss += self.best_loss[item]

        if best_loss < self.stop_loss:
            self.update_c = 0
            self.logger.info(
                f"[{self.update_c}/{self.args.stop_early}] update ==> Total loss {best_loss:.03f}"
            )
        else:
            self.update_c += 1
            self.logger.info(
                f"[{self.update_c}/{self.args.stop_early}] Nothing do not update  ==> Total loss {best_loss:.03f}"
            )

    def reset_log(self, mode):
        self.test_value = (
            {
                "sagging": AverageMeter(),
                "wrinkle": AverageMeter(),
                "pore": AverageMeter(),
                "pigmentation": AverageMeter(),
                "dryness": AverageMeter(),
            }
            if mode == "class"
            else {
                "moisture": AverageMeter(),
                "wrinkle": AverageMeter(),
                "elasticity": AverageMeter(),
                "pore": AverageMeter(),
                "count": AverageMeter(),
            }
        )

        self.train_loss = copy.deepcopy(self.test_value)
        self.val_loss = copy.deepcopy(self.test_value)
        self.pred = list()
        self.gt = list()

    def update_e(self, epoch):
        self.epoch = epoch
        self.stop_loss = sum(self.args.best_loss.values())

    def get_test_acc(self, pred, label):
        gt = (
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
            .permute(1, 0)
            .to(device)
        )
        num = 0

        for idx, area_name in enumerate(label):
            dig = area_name.split("_")[-1]
            pred_l = torch.argmax(pred[:, num : num + class_num_list[dig]], dim=1)
            num += class_num_list[dig]
            self.test_value[dig].update_acc(
                (pred_l == gt[:, idx]).sum().item(),
                pred_l.shape[0],
            )

    def get_test_loss(self, pred_p, label, area_num):
        count = 0
        for name in self.equip_loss[area_num]:
            gt = label[:, count : count + self.equip_loss[area_num][name]]
            pred = pred_p[:, count : count + self.equip_loss[area_num][name]]
            count += self.equip_loss[area_num][name]
            self.test_value[name].update(
                self.criterion(pred, gt).item(), batch_size=pred.shape[0]
            )

    def run(self, dig, num_class, phase="train"):
        self.phase = phase
        self.model = (
            copy.deepcopy(self.model_list[self.m_dig])
            if self.phase == "train"
            else self.temp_model_list[self.m_dig]
        )
        self.model.train() if self.phase == "train" else self.model.eval()

        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )

        adjust_learning_rate(optimizer, self.epoch, self.args)
        data_loader = self.train_loader if self.phase == "train" else self.valid_loader

        def run_iter():
            total_iter = 0
            random_num = random.randrange(1, len(data_loader))
            for patch_list in data_loader:
                for item in patch_list[dig]:
                    img, label = get_item(item, device)
                    pred = pred_image(self, img, item[2])

                    if self.args.mode == "class":
                        loss = self.class_loss(pred, label, dig, num_class)
                    else:
                        loss = self.regression(pred, label, dig)

                    total_iter += 1
                    self.print_loss(total_iter, len(patch_list[dig]))
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                if total_iter == random_num and dig == "wrinkle":
                    save_image(self, patch_list, self.epoch)   
                
                    

            self.print_loss(total_iter, len(patch_list[dig]), final_flag=True)

        if phase == "train":
            run_iter()
        else:
            with torch.no_grad():
                run_iter()
