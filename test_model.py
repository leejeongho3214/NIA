import errno
import os
import cv2
import torch
import cv2
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
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


class Model_test(object):
    def __init__(self, args, model_list, valid_loader):
        super(Model_test, self).__init__()
        self.args = args
        self.model_list = model_list
        self.temp_model_list = [None for _ in range(8)]
        self.valid_loader = valid_loader
        self.best_acc = args.best_loss.copy()
        self.best_loss = args.best_loss.copy()

        self.val_acc = [AverageMeter() for _ in range(8)]
        self.log_loss = [AverageMeter() for _ in range(8)]
        self.val_loss = [AverageMeter() for _ in range(8)]
        self.log_loss_test = {
            "moisture": AverageMeter(),
            "wrinkle": [AverageMeter() for _ in range(8)],
            "elasticity": [AverageMeter() for _ in range(14)],
            "pore": AverageMeter(),
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
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0
        self.logger = None

    def choice(self, m_idx):
        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def print_total(self):
        if self.args.mode == "class":
            print(
                f"[{self.phase}] [Not Update -> {self.update_c}] pigmentation: {(self.log_acc['pigmentation'].avg * 100):.2f}% // wrinkle: {(self.log_acc['wrinkle'].avg * 100):.2f}% // sagging: {(self.log_acc['sagging'].avg * 100):.2f}% // pore: {(self.log_acc['pore'].avg * 100):.2f}% // dryness: {(self.log_acc['dryness'].avg * 100):.2f}%"
            )
            if self.logger is not None:
                self.logger.debug(
                    f"Epoch: {self.epoch} [Not Update -> {self.update_c}] [{self.phase}] pigmentation: {(self.log_acc['pigmentation'].avg * 100):.2f}% // wrinkle: {(self.log_acc['wrinkle'].avg * 100):.2f}% // sagging: {(self.log_acc['sagging'].avg * 100):.2f}% // pore: {(self.log_acc['pore'].avg * 100):.2f}% // dryness: {(self.log_acc['dryness'].avg * 100):.2f}%"
                )
        else:
            print(
                f"[{self.phase}] [Not Update -> {self.update_c}] moisture: {(self.log_loss_test['moisture'].avg)}// wrinkle: {(self.log_loss_test['wrinkle'].avg)}// elasticity: {(self.log_loss_test['elasticity'].avg)}// pore: {(self.log_loss_test['pore'].avg)}"
            )
            if self.logger is not None:
                self.logger.debug(
                    f"Epoch: {self.epoch} [Not Update -> {self.update_c}] [{self.phase}] moisture: {(self.log_loss_test['moisture'].avg)}// wrinkle: {(self.log_loss_test['wrinkle'].avg)}// elasticity: {(self.log_loss_test['elasticity'].avg)} // pore: {(self.log_acc['pore'].avg)}"
                )

    def print_test(self):
        print(
            f"[{self.phase}] [Not Update -> {self.update_c}] moisture: {(self.log_loss_test['moisture'].avg * 1):0.3f}// pore: {int(self.log_loss_test['pore'].avg * 3000)}"
        )

        # wrinkle_scale = [100, 100, 300, 100, 300, 200, 200, 200]
        # elascity_scale = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]

        wrinkle_scale = [1 for _ in range(8)]
        elascity_scale = [1 for _ in range(14)]

        for i in range(len(self.log_loss_test["wrinkle"])):
            print(
                f"wrinkle_{i} ==> {(self.log_loss_test['wrinkle'][i].avg * wrinkle_scale[i]):0.3f}"
            )

        for i in range(len(self.log_loss_test["elasticity"])):
            print(
                f"elasticity{i} ==> {(self.log_loss_test['elasticity'][i].avg * elascity_scale[i]):0.3f}"
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

    def nan_detect(self, label):
        nan_list = list()
        for batch_idx, batch_data in enumerate(label):
            for value in batch_data:
                if not torch.isfinite(value):
                    nan_list.append(batch_idx)
        return nan_list

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
            if self.equip_loss[area_num][name] > 1:
                for i in range(self.equip_loss[area_num][name]):
                    gt = label[:, count : count + 1]
                    pred = pred_p[:, count : count + 1]
                    self.log_loss_test[name][i].update_train(
                        self.criterion(pred, gt).item(), batch_size=pred.shape[0]
                    )
                    count += 1
            else:
                gt = label[:, count : count + self.equip_loss[area_num][name]]
                pred = pred_p[:, count : count + self.equip_loss[area_num][name]]
                count += self.equip_loss[area_num][name]
                self.log_loss_test[name].update_train(
                    self.criterion(pred, gt).item(), batch_size=pred.shape[0]
                )

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
