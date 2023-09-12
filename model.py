import errno
import os
import cv2
import torch
import cv2
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from data_loader import folder_name, class_num_list, area_naming


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


class Model(object):
    def __init__(self, args, model_list, train_loader, valid_loader, logger, writer):
        super(Model, self).__init__()
        self.args = args
        self.model_list = model_list
        self.temp_model_list = [None for _ in range(8)]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.best_acc = args.best_loss.copy()
        self.best_loss = args.best_loss.copy()

        self.writer = writer
        self.logger = logger

        self.val_acc = [AverageMeter() for _ in range(8)]
        self.log_loss = [AverageMeter() for _ in range(8)]
        self.val_loss = [AverageMeter() for _ in range(8)]
        self.log_loss_test = {
            "moisture": AverageMeter(),
            "wrinkle": AverageMeter(),
            "elasticity": AverageMeter(),
            "pore": AverageMeter()
        }
        self.log_acc = {
            "moisture": AverageMeter(),
            "elasticity": AverageMeter(),
            "wrinkle": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation": AverageMeter(),
        }

        self.equip_loss = {
            "1": {"moisture": 1, "elasticity": 14},
            "3": {"wrinkle": 8},
            "4": {"wrinkle": 8},
            "5": {"moisture": 1, "elasticity": 14, "pore": 1},
            "6": {"moisture": 1, "elasticity": 14, "pore": 1},
            "8": {"moisture": 1, "elasticity": 14},
        }

        self.epoch = 0
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.MSELoss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0

    def choice(self, m_idx):
        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def update_m(self, model_num_class):
        count = 0
        if self.args.mode == "class":
            for idx in range(8):
                if self.best_acc[idx] < self.val_acc[idx].avg:
                    self.best_acc[idx] = self.val_acc[idx].avg
                    self.model_list[idx] = copy.deepcopy(self.temp_model_list[idx])
                    save_checkpoint(
                        self.model_list[idx], self.args, self.epoch, idx, self.best_acc
                    )
                    count += 1
        else:
            for idx, value in enumerate(model_num_class):
                if np.isnan(value):
                    continue
                if self.best_loss[idx] > self.val_loss[idx].avg:
                    self.best_loss[idx] = self.val_loss[idx].avg
                    self.model_list[idx] = copy.deepcopy(self.temp_model_list[idx])
                    save_checkpoint(
                        self.model_list[idx], self.args, self.epoch, idx, self.best_loss
                    )
                    count += 1
                    
        if count == 0:
            self.update_c += 1
        
        else: 
            self.update_c = 0

    def update_e(self, epoch):
        self.epoch = epoch

    def print_total(self):
        if self.args.mode == 'class':
            print(
                f"[{self.phase}] [Not Update -> {self.update_c}] pigmentation: {(self.log_acc['pigmentation'].avg * 100):.2f}% // wrinkle: {(self.log_acc['wrinkle'].avg * 100):.2f}% // sagging: {(self.log_acc['sagging'].avg * 100):.2f}% // pore: {(self.log_acc['pore'].avg * 100):.2f}% // dryness: {(self.log_acc['dryness'].avg * 100):.2f}%"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Not Update -> {self.update_c}] [{self.phase}] pigmentation: {(self.log_acc['pigmentation'].avg * 100):.2f}% // wrinkle: {(self.log_acc['wrinkle'].avg * 100):.2f}% // sagging: {(self.log_acc['sagging'].avg * 100):.2f}% // pore: {(self.log_acc['pore'].avg * 100):.2f}% // dryness: {(self.log_acc['dryness'].avg * 100):.2f}%"
            )
        else:

            print(
                f"[{self.phase}] [Not Update -> {self.update_c}] moisture: {(self.log_loss_test['moisture'].avg)}// wrinkle: {(self.log_loss_test['wrinkle'].avg)}// elasticity: {(self.log_loss_test['elasticity'].avg)}// pore: {(self.log_loss_test['pore'].avg)}"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Not Update -> {self.update_c}] [{self.phase}] moisture: {(self.log_loss_test['moisture'].avg)}// wrinkle: {(self.log_loss_test['wrinkle'].avg)}// elasticity: {(self.log_loss_test['elasticity'].avg)} // pore: {(self.log_acc['pore'].avg)}"
            )

    def train_print(self, iteration):
        if iteration == len(self.train_loader) - 1:
            print(
                f"\rEpoch: {self.epoch} [Train][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {self.log_loss[self.m_idx].avg}"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Train][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {self.log_loss[self.m_idx].avg}"
            )
            self.writer.add_scalar(
                f"train/{area_naming[f'{self.m_idx}']}",
                self.log_loss[self.m_idx].avg,
                self.epoch,
            )
            self.temp_model_list[self.m_idx] = self.model

        else:
            print(
                f"\rEpoch: {self.epoch} [Train][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {self.log_loss[self.m_idx].avg}",
                end="",
            )

    def valid_acc_print(self, iteration):
        if iteration == len(self.valid_loader) - 1:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.val_acc[self.m_idx].avg * 100):.2f}%"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Val][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.val_acc[self.m_idx].avg * 100):.2f}%"
            )
            self.writer.add_scalar(
                f"val/{area_naming[f'{self.m_idx}']}",
                self.val_acc[self.m_idx].avg * 100,
                self.epoch,
            )

        else:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.val_acc[self.m_idx].avg * 100):.2f}%",
                end="",
            )

    def valid_loss_print(self, iteration):
        if iteration == len(self.valid_loader) - 1:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- > loss: {self.val_loss[self.m_idx].avg}"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Val][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- > loss: {self.val_loss[self.m_idx].avg}"
            )
            self.writer.add_scalar(
                f"val/{area_naming[f'{self.m_idx}']}",
                self.val_loss[self.m_idx].avg,
                self.epoch,
            )

        else:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_naming[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- > loss: {self.val_loss[self.m_idx].avg}",
                end="",
            )

    def reset_log(self):
        self.val_acc = [AverageMeter() for _ in range(8)]
        self.log_loss = [AverageMeter() for _ in range(8)]
        self.val_loss = [AverageMeter() for _ in range(8)]
        self.log_loss_test = {
            "moisture": AverageMeter(),
            "wrinkle": AverageMeter(),
            "elasticity": AverageMeter(),
            "pore": AverageMeter()
        }
        self.log_acc = {
            "moisture": AverageMeter(),
            "elasticity": AverageMeter(),
            "wrinkle": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation": AverageMeter(),
        }


    def class_loss(self, pred, label):
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

        self.log_loss[self.m_idx].update_train(
            loss, batch_size=gt.shape[0]
        )  ## check -> batch_size is correct?

        return loss

    def regression(self, pred, label):
        loss = self.criterion(pred, label)
        if self.phase == "train":
            self.log_loss[self.m_idx].update_train(loss, batch_size=pred.shape[0])
        else:
            self.val_loss[self.m_idx].update_train(loss, batch_size=pred.shape[0])

        return loss

    def save_img(self, iteration, patch_list):
        if iteration == 0 and self.epoch == 0:
            num_patch = len(patch_list)
            vis_img = np.zeros([128 * 2, 128 * (num_patch // 2), 3])

            for idx, area_num in enumerate(patch_list):
                col = idx % (num_patch // 2)
                row = idx // (num_patch // 2)
                vis_img[
                    row * 128 : (row + 1) * 128, col * 128 : (col + 1) * 128
                ] = patch_list[area_num][0][0].permute(1, 2, 0)
            mkdir(f"vis/{self.args.mode}/{self.args.name}/{self.m_idx}")
            cv2.imwrite(
                f"vis/{self.args.mode}/{self.args.name}/{self.m_idx}/epoch_{self.epoch}.jpg",
                vis_img * 256,
            )

    def nan_detect(self, label):
        nan_list = list()
        for batch_idx, batch_data in enumerate(label):
            for value in batch_data:
                if not torch.isfinite(value):
                    nan_list.append(batch_idx)
        return nan_list

    def train(self):
        self.model = copy.deepcopy(self.model_list[self.m_idx])
        self.model.train()

        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        self.phase = "train"
        area_num = str(self.m_idx + 1)
        for iteration, patch_list in enumerate(self.train_loader):
            img, label = patch_list[area_num][0].to(device), patch_list[area_num][1].to(
                device
            )

            adjust_learning_rate(optimizer, self.epoch, self.args)
            pred = self.model.to(device)(img)

            if self.args.mode == "class":
                loss = self.class_loss(pred, label)

            else:
                idx_list = np.array([idx for idx in range(label.size(0))])
                nan_list = self.nan_detect(label)
                idx_list = idx_list[idx_list != nan_list]
                if len(idx_list) > 0:
                    loss = self.regression(pred[idx_list], label[idx_list])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.save_img(iteration, patch_list)

            self.train_print(iteration)

    def get_val_acc(self, pred, label):
        gt = (
            torch.tensor(np.array([label[value].numpy() for value in label]))
            .permute(1, 0)
            .to(device)
        )

        num = 0
        count = 0
        for idx, area_name in enumerate(label):
            pred_l = torch.argmax(pred[:, num : num + class_num_list[area_name]], dim=1)
            num += class_num_list[area_name]
            count += (pred_l == gt[:, idx]).sum().item()

        self.val_acc[self.m_idx].update_val(count, gt.shape[0] * gt.shape[1])

    def get_test_acc(self, pred, label):
        gt = (
            torch.tensor(np.array([label[value].numpy() for value in label]))
            .permute(1, 0)
            .to(device)
        )
        num = 0

        for idx, area_name in enumerate(label):
            pred_l = torch.argmax(pred[:, num : num + class_num_list[area_name]], dim=1)
            num += class_num_list[area_name]
            self.log_acc[area_name].update_val(
                (pred_l == gt[:, idx]).sum().item(),
                pred_l.shape[0],
            )

    def get_test_loss(self, pred_p, label, area_num):
        count = 0
        for name in self.equip_loss[area_num]:
            gt = label[:, count : count + self.equip_loss[area_num][name]]
            pred = pred_p[:, count : count + self.equip_loss[area_num][name]]
            count += self.equip_loss[area_num][name]
            self.log_loss_test[name].update_train(
                self.criterion(pred, gt).item(), batch_size=pred.shape[0]
            )
            
            if np.isnan(self.log_loss_test[name].avg):
                pass


    def valid(self):
        self.phase = "valid"
        self.model = self.temp_model_list[self.m_idx]
        self.model.eval()
        area_num = str(self.m_idx + 1)
        with torch.no_grad():
            for iteration, patch_list in enumerate(self.valid_loader):
                img, label = (
                    patch_list[area_num][0].to(device),
                    patch_list[area_num][1].to(device),
                )

                pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    self.get_val_acc(pred, label)
                    self.valid_acc_print(iteration)

                else:
                    idx_list = np.array([idx for idx in range(label.size(0))])
                    nan_list = self.nan_detect(label)
                    idx_list = idx_list[idx_list != nan_list]
                    if len(idx_list) > 0:
                        self.regression(pred[idx_list], label[idx_list])
                        self.valid_loss_print(iteration)
                    else:
                        self.valid_loss_print(iteration)


    def test(self):
        self.phase = "test"
        self.model = copy.deepcopy(self.model_list[self.m_idx])
        area_num = str(self.m_idx + 1)
        self.model.eval()
        with torch.no_grad():
            for _, patch_list in enumerate(self.valid_loader):
                img, label = (
                    patch_list[area_num][0].to(device),
                    patch_list[area_num][1].to(device),
                )

                pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    self.get_test_acc(pred, label)
                else:
                    idx_list = np.array([idx for idx in range(label.size(0))])
                    nan_list = self.nan_detect(label)
                    idx_list = idx_list[idx_list != nan_list]
                    if len(idx_list) > 0:
                        self.get_test_loss(pred[idx_list], label[idx_list], area_num)
                
