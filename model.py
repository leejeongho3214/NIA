import errno
import os
import cv2
import torch
import cv2
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from data_loader import folder_name, class_num_list, area_naming, area_name


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


class Model(object):
    def __init__(
        self,
        args,
        model_list,
        train_loader,
        valid_loader,
        testset_loader,
        logger,
        writer,
    ):
        super(Model, self).__init__()
        self.args = args
        self.model_list = model_list
        self.temp_model_list = [None for _ in range(8)]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = testset_loader
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
            "pore": AverageMeter(),
        }
        self.log_acc = {
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

        self.keep_loss = {
            "moisture": 0,
            "wrinkle": 0,
            "elasticity": 0,
            "pore": 0,
        }

        self.equip_loss = {
            "1": {"moisture": 1, "elasticity": 1},
            "3": {"wrinkle": 1},
            "4": {"wrinkle": 1},
            "5": {"moisture": 1, "elasticity": 1, "pore": 1},
            "6": {"moisture": 1, "elasticity": 1, "pore": 1},
            "8": {"moisture": 1, "elasticity": 1},
        }

        self.epoch = 0
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0

    def choice(self, m_idx):
        if m_idx in [3, 5]:
            m_idx -= 1
            self.flag = True    
        else:
            self.flag = False
        
        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def update_m(self, model_num_class):
        count = 0
        if self.args.mode == "class":
            for idx in range(8):
                if idx in [3, 5]: continue
                if self.best_acc[idx] < self.val_acc[idx].avg:
                    self.best_acc[idx] = round(self.val_acc[idx].avg, 2)
                    self.model_list[idx] = copy.deepcopy(self.temp_model_list[idx])
                    save_checkpoint(
                        self.model_list[idx], self.args, self.epoch, idx, self.best_acc
                    )
                    count += 1
        else:
            for idx, value in enumerate(model_num_class):
                if np.isnan(value) or idx in [3, 5]:
                    continue
                if self.best_loss[idx] > self.val_loss[idx].avg:
                    self.best_loss[idx] = round(self.val_loss[idx].avg.item(), 4)
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

    def acc_avg(self, name):
        return round(self.log_acc[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.log_loss_test[name].avg, 4)

    def up_and_down(self, name, color="\033[95m", c_color="\033[0m"):
        if self.args.mode == "class":
            sub = (self.log_acc[name].avg * 100) - self.keep_acc[name]
            value = round(sub, 2)
            result = (
                f"{color}+{value}{c_color}%"
                if value > 0
                else "No change"
                if value == 0
                else f"{color}{value}{c_color}%"
            )
        else:
            sub = (self.log_loss_test[name].avg) - self.keep_loss[name]
            value = round(sub, 4)
            result = (
                f"{color}+{value}{c_color}"
                if value > 0
                else "No change"
                if value == 0
                else f"{color}{value}{c_color}"
            )

        return result

    def print_total(self):
        if self.args.mode == "class":
            print(
                f"[{self.phase}] [Early Stop: {self.update_c}/{self.args.stop_early}] pigmentation: {self.acc_avg('pigmentation')}%({self.up_and_down('pigmentation')}) // wrinkle: {self.acc_avg('wrinkle')}%({self.up_and_down('wrinkle')}) // sagging: {self.acc_avg('sagging')}%({self.up_and_down('sagging')}) // pore: {self.acc_avg('pore')}%({self.up_and_down('pore')}) // dryness: {self.acc_avg('dryness')}%({self.up_and_down('dryness')})"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Early Stop: {self.update_c}/{self.args.stop_early}] pigmentation: {self.acc_avg('pigmentation')}%({self.up_and_down('pigmentation', color ='', c_color='')}) // wrinkle: {self.acc_avg('wrinkle')}%({self.up_and_down('wrinkle', color ='', c_color='')}) // sagging: {self.acc_avg('sagging')}%({self.up_and_down('sagging', color ='', c_color='')}) // pore: {self.acc_avg('pore')}%({self.up_and_down('pore', color ='', c_color='')}) // dryness: {self.acc_avg('dryness')}%({self.up_and_down('dryness', color ='', c_color='')})"
            )
            for name in self.log_acc:
                self.keep_acc[name] = self.log_acc[name].avg * 100
        else:
            print(
                f"[{self.phase}] [Early Stop: {self.update_c}/{self.args.stop_early}] moisture: {self.loss_avg('moisture')}({self.up_and_down('moisture')}) // wrinkle: {self.loss_avg('wrinkle')}({self.up_and_down('wrinkle')}) // elasticity: {self.loss_avg('elasticity')}({self.up_and_down('elasticity')}) // pore: {self.loss_avg('pore')}({self.up_and_down('pore')})"
            )
            self.logger.debug(
                f"Epoch: {self.epoch} [Early Stop: {self.update_c}/{self.args.stop_early}] moisture: {self.loss_avg('moisture')}({self.up_and_down('moisture', color = '', c_color='')}) // wrinkle: {self.loss_avg('wrinkle')}({self.up_and_down('wrinkle', color = '', c_color='')}) // elasticity: {self.loss_avg('elasticity')}({self.up_and_down('elasticity', color = '', c_color='')}) // pore: {self.loss_avg('pore')} ({self.up_and_down('pore', color = '', c_color='')})"
            )
            for name in self.log_loss_test:
                self.keep_loss[name] = self.log_loss_test[name].avg

    def train_print(self, iteration):
        if iteration == len(self.train_loader) - 1:
            print(
                f"\rEpoch: {self.epoch} [Train][{area_name[f'{self.m_idx}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {(self.log_loss[self.m_idx].avg):.04f}"
            )

            self.writer.add_scalar(
                f"train/{area_name[f'{self.m_idx}']}",
                self.log_loss[self.m_idx].avg,
                self.epoch,
            )
            self.temp_model_list[self.m_idx] = self.model

        else:
            print(
                f"\rEpoch: {self.epoch} [Train][{area_name[f'{self.m_idx}']}][{iteration}/{len(self.train_loader)}] ---- >  loss: {(self.log_loss[self.m_idx].avg):.04f}",
                end="",
            )

    def valid_acc_print(self, iteration):
        if iteration == len(self.valid_loader) - 1:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_name[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.val_acc[self.m_idx].avg * 100):.2f}%"
            )

            self.writer.add_scalar(
                f"val/{area_name[f'{self.m_idx}']}",
                self.val_acc[self.m_idx].avg * 100,
                self.epoch,
            )

        else:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_name[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- >  acc: {(self.val_acc[self.m_idx].avg * 100):.2f}%",
                end="",
            )

    def valid_loss_print(self, iteration):
        if iteration == len(self.valid_loader) - 1:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_name[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- > loss: {(self.val_loss[self.m_idx].avg):.04f}"
            )
            self.writer.add_scalar(
                f"val/{area_name[f'{self.m_idx}']}",
                self.val_loss[self.m_idx].avg,
                self.epoch,
            )

        else:
            print(
                f"\rEpoch: {self.epoch} [Val][{area_name[f'{self.m_idx}']}][{iteration}/{len(self.valid_loader)}] ---- > loss: {(self.val_loss[self.m_idx].avg):.04f}",
                end="",
            )

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            return True

    def reset_log(self):
        self.val_acc = [AverageMeter() for _ in range(8)]
        self.log_loss = [AverageMeter() for _ in range(8)]
        self.val_loss = [AverageMeter() for _ in range(8)]
        self.log_loss_test = {
            "moisture": AverageMeter(),
            "wrinkle": AverageMeter(),
            "elasticity": AverageMeter(),
            "pore": AverageMeter(),
        }
        self.log_acc = {
            "sagging": AverageMeter(),
            "wrinkle": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation": AverageMeter(),
            "dryness": AverageMeter(),
        }

    def class_loss(self, pred, label):
        num = 0
        loss = 0
        for name in label:
            gt = F.one_hot(
                label[name].type(torch.int64), num_classes=class_num_list[name]
            ).to(device)
            pred_p = softmax(pred[:, num : num + class_num_list[name]])
            num += class_num_list[name]
            loss += self.criterion(pred_p.type(torch.float), gt.type(torch.float))

        self.log_loss[self.m_idx].update_train(
            loss, batch_size=gt.shape[0]
        )  ## check -> batch_size is correct?

        return loss

    def regression(self, pred, label):
        loss = self.criterion(pred, label.to(device))
        if self.phase == "train":
            self.log_loss[self.m_idx].update_train(loss, batch_size=pred.shape[0])
        else:
            self.val_loss[self.m_idx].update_train(loss, batch_size=pred.shape[0])

        return loss

    def match_img(self, vis_img, img):
        col = self.num % (self.num_patch // 4)
        row = self.num // (self.num_patch // 4)
        vis_img[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = img

        return vis_img

    def save_img(self, iteration, patch_list):
        if iteration == 0 and self.epoch == 0:
            self.num_patch = len(patch_list) + 5
            if self.args.mode == "class":
                vis_img = np.zeros([256 * 5, 256 * 3, 3])
            else:
                vis_img = np.zeros([256 * 5, 256 * 2, 3])

            self.num = 0
            for area_num in patch_list:
                img = patch_list[area_num][0][0].permute(1, 2, 0)
                # if self.args.normalize:
                #     img = (img+1)/2

                if int(area_num) in [1, 7, 8]:
                    l_img = (img[:, :128]).numpy()
                    l_img = cv2.resize(l_img, (256, 256))
                    cv2.putText(
                        l_img,
                        f"{area_naming[str(int(area_num)-1)]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, l_img)
                    self.num += 1
                    r_img = (img[:, 128:]).numpy()
                    r_img = cv2.resize(r_img, (256, 256))
                    cv2.putText(
                        r_img,
                        f"{area_naming[str(int(area_num)-1)]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, r_img)
                    self.num += 1

                elif int(area_num) in [3, 4]:
                    l_img = (img[:128]).numpy()
                    l_img = cv2.resize(l_img, (256, 256))
                    cv2.putText(
                        l_img,
                        f"{area_naming[str(int(area_num)-1)]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, l_img)
                    self.num += 1
                    r_img = (img[128:]).numpy()
                    r_img = cv2.resize(r_img, (256, 256))
                    cv2.putText(
                        r_img,
                        f"{area_naming[str(int(area_num)-1)]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, r_img)
                    self.num += 1

                else:
                    img = (img[:, :128]).numpy()
                    img = cv2.resize(img, (256, 256))
                    cv2.putText(
                        img,
                        f"{area_naming[str(int(area_num)-1)]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, img)
                    self.num += 1

            mkdir(f"vis/{self.args.mode}/{self.args.name}/{self.m_idx}")
            cv2.imwrite(
                f"vis/{self.args.mode}/{self.args.name}/{self.m_idx}/epoch_{self.epoch}.jpg",
                vis_img ,
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
        area_num = str(self.m_idx + 2) if self.flag else str(self.m_idx + 1)
        for iteration, patch_list in enumerate(self.train_loader):
            if type(patch_list[area_num][1]) == torch.Tensor:
                label = patch_list[area_num][1].to(device)
            else:
                for name in patch_list[area_num][1]:
                    patch_list[area_num][1][name] = patch_list[area_num][1][name].to(
                        device
                    )
                label = patch_list[area_num][1]

            img = patch_list[area_num][0].to(device)
            adjust_learning_rate(optimizer, self.epoch, self.args)

            if area_num in [1, 7, 8]:
                img_l = img[:, :, :, :128]
                img_r = img[:, :, :, 128:]
                pred = self.model.to(device)(img_l)
                pred = self.model.to(device)(img_r) + pred

            elif area_num in [3, 4]:
                img_l = img[:, :, :128, :]
                img_r = img[:, :, 128:, :]
                pred = self.model.to(device)(img_l)
                pred = self.model.to(device)(img_r) + pred

            else:
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
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
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
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
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

    def valid(self):
        self.phase = "valid"
        self.model = self.temp_model_list[self.m_idx]
        self.model.eval()
        area_num = str(self.m_idx + 2) if self.flag else str(self.m_idx + 1)
        with torch.no_grad():
            for iteration, patch_list in enumerate(self.valid_loader):
                img, label = (
                    patch_list[area_num][0].to(device),
                    patch_list[area_num][1],
                )

                if area_num in [1, 7, 8]:
                    img_l = img[:, :, :, :128]
                    img_r = img[:, :, :, 128:]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                elif area_num in [3, 4]:
                    img_l = img[:, :, :128, :]
                    img_r = img[:, :, 128:, :]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                else:
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
        area_num = str(self.m_idx + 2) if self.flag else str(self.m_idx + 1)
        self.model.eval()
        with torch.no_grad():
            for _, patch_list in enumerate(self.test_loader):
                img, label = (
                    patch_list[area_num][0].to(device),
                    patch_list[area_num][1],
                )

                if area_num in [1, 7, 8]:
                    img_l = img[:, :, :, :128]
                    img_r = img[:, :, :, 128:]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                elif area_num in [3, 4]:
                    img_l = img[:, :, :128, :]
                    img_r = img[:, :, 128:, :]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                else:
                    pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    self.get_test_acc(pred, label)
                else:
                    idx_list = np.array([idx for idx in range(label.size(0))])
                    nan_list = self.nan_detect(label)
                    idx_list = idx_list[idx_list != nan_list]
                    if len(idx_list) > 0:
                        self.get_test_loss(
                            pred[idx_list], label[idx_list].to(device), area_num
                        )
