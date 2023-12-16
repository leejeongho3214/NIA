from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
import copy
from data_loader import class_num_list
from utils import LabelSmoothingCrossEntropy, get_item, labeling, pred_image, AverageMeter, save_checkpoint, save_image
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    return e_x / torch.sum(e_x, dim=1).unsqueeze(dim=1)


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.epoch / 2.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def resume_checkpoint(args, model, path):
    state_dict = torch.load(path, map_location=device)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict
    args.load_epoch = epoch
    args.best_loss = best_loss

    return model


class Model(object):
    def __init__(
        self,
        args,
        model_list,
        train_loader,
        valid_loader,
        logger,
        writer,
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
            self.writer,
            self.logger,
            self.check_path,
        ) = (
            args,
            model_list,
            defaultdict(str),
            train_loader,
            valid_loader,
            args.best_loss,
            writer,
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
            LabelSmoothingCrossEntropy()
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
            self.writer.add_scalar(
                f"{self.phase}/{self.m_dig}",
                self.train_loss[self.m_dig].avg
                if self.args.mode == "class"
                else self.val_loss[self.m_dig].avg,
                self.epoch,
            )
            self.temp_model_list[self.m_dig] = self.model

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            return True

    def class_loss(self, pred, label, dig):
        loss = self.criterion(pred, label, smoothing = self.args.smooth)
        pred_p = softmax(pred)
        
        self.pred.append([dig, pred_p.argmax().item()])
        self.gt.append([dig, label.item()])

        self.train_loss[self.m_dig].update(
            loss, batch_size= 1
        ) if self.phase == "train" else self.val_loss[self.m_dig].update(
            loss, batch_size= 1
        )

        return loss

    def regression(self, pred, label, dig):
        loss = self.criterion(pred[0], label.to(device))
        
        self.pred.append([dig, pred[0].item()])
        self.gt.append([dig, label.item()])
        self.train_loss[self.m_dig].update(
            loss, batch_size=pred.shape[0]
        ) if self.phase == "train" else self.val_loss[self.m_dig].update(
            loss, batch_size=pred.shape[0]
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
            os.path.join(self.check_path, f"epoch_{self.epoch}_pred.txt"), "w"
        ) as p:
            with open(
                os.path.join(self.check_path, f"epoch_{self.epoch}_gt.txt"), "w"
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

    def run(self, dig, phase="train"):
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
            for patch_list in data_loader:
                for item in patch_list[dig]:
                    img, label = get_item(item, device)
                    pred = pred_image(self, img, item[2])

                    if self.args.mode == "class":
                        loss = self.class_loss(pred, label, dig)
                    else:
                        loss = self.regression(pred, label, dig)

                    total_iter += 1
                    self.print_loss(total_iter, len(patch_list[dig]))
                    
                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    if total_iter == 1 and dig == "wrinkle":
                        save_image(self, patch_list, self.epoch)        
                
                    

            self.print_loss(total_iter, len(patch_list[dig]), final_flag=True)

        if phase == "train":
            run_iter()
        else:
            with torch.no_grad():
                run_iter()
