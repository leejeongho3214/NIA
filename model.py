import errno
import gc
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
        self.args = args
        self.model_list = model_list
        self.temp_model_list = [None for _ in range(9)]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.best_loss = args.best_loss

        self.writer = writer
        self.logger = logger
        self.check_path = check_path

        self.train_loss = [AverageMeter() for _ in range(9)]
        self.val_loss = [AverageMeter() for _ in range(9)]

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
            nn.CrossEntropyLoss(label_smoothing=self.args.label_smooth) if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0
        self.stop_loss = np.inf
        
        self.pred = list()
        self.gt = list()

    def choice(self, m_idx):
        if m_idx in [4, 6]:
            m_idx -= 1
            self.flag = True
        else:
            self.flag = False

        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def update_m(self, model_num_class):
        best_loss = 0
        for idx, value in enumerate(model_num_class):
            if np.isnan(value) or idx in [4, 6]:
                continue
            
            if self.best_loss[idx] > self.val_loss[idx].avg:
                try:
                    self.best_loss[idx] = round(self.val_loss[idx].avg.item(), 4)
                except:
                    self.best_loss[idx] = round(self.val_loss[idx].avg, 4)
                    
                self.model_list[idx] = copy.deepcopy(self.temp_model_list[idx])
                save_checkpoint(
                    self.model_list[idx], self.args, self.epoch, idx, self.best_loss
                )
                
            best_loss += self.best_loss[idx]

        if best_loss < self.stop_loss:
            self.update_c = 0
            self.logger.info(f"[{self.update_c}/{self.args.stop_early}] update ==> Total loss {best_loss:.03f}")
            
        else:
            self.update_c += 1
            self.logger.info(f"[{self.update_c}/{self.args.stop_early}] Nothing do not update  ==> Total loss {best_loss:.03f}")

    def update_e(self, epoch):
        self.epoch = epoch
        self.stop_loss = sum(self.args.best_loss)
    
    def save_value(self):   
        with open(os.path.join(self.check_path, f"epoch_{self.epoch}_pred.txt"), "w") as p:
            with open(os.path.join(self.check_path, f"epoch_{self.epoch}_gt.txt"), "w") as g:
                for idx in range(len(self.pred)):
                    p.write(f"{self.pred[idx][0]}, {self.pred[idx][1]} \n")
                    g.write(f"{self.gt[idx][0]}, {self.gt[idx][1]} \n")
        g.close()
        p.close()

    def acc_avg(self, name):
        return round(self.test_value[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_value[name].avg, 4)

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


    def print_loss(self, iteration):
        dataloader_len = (
            len(self.train_loader) if self.phase == "train" else len(self.valid_loader)
        )

        if iteration == dataloader_len - 1:
            self.logger.info(f"Epoch: {self.epoch} [{self.phase}][{area_naming[f'{self.area_num}']}][{iteration}/{dataloader_len}] ---- >  loss: {self.train_loss[self.m_idx].avg if self.phase == 'train' else self.val_loss[self.m_idx].avg:.04f}")
            self.writer.add_scalar(
                f"{self.phase}/{area_naming[f'{self.m_idx}']}",
                self.train_loss[self.m_idx].avg
                if self.args.mode == "class"
                else self.val_loss[self.m_idx].avg,
                self.epoch,
            )
            self.temp_model_list[self.m_idx] = self.model

        else:
            print(
                f"\rEpoch: {self.epoch} [{self.phase}][{area_naming[f'{self.area_num}']}][{iteration}/{dataloader_len}] ---- >  loss: {self.train_loss[self.m_idx].avg if self.phase == 'train' else self.val_loss[self.m_idx].avg:.04f}",
                end="",
            )

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            return True

    def reset_log(self, mode):
        self.train_loss = [AverageMeter() for _ in range(9)]
        self.val_loss = [AverageMeter() for _ in range(9)]
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
        self.pred = list()
        self.gt = list()

    def labeling(self, label, num, bs):
        template = torch.zeros(num)
        gt = label.item()
        if gt == 0:
            template[0] = 0.6
            template[1] = 0.3
            template[2] = 0.1
        
        elif gt == num - 1:
            template[-1] = 0.6
            template[-2] = 0.3
            template[-3] = 0.1
            
        else:
            template[gt] = 0.5
            template[gt - 1] = 0.25
            template[gt + 1] = 0.25 
            
        return template.reshape(1, -1)
    
    def class_loss(self, pred, label):
        num = 0
        loss = 0
        bs = pred.shape[0]
        for name in label:
            area, dig = name.split("_")[-2:]
            class_num = 9 if area == "forehead" and dig == "wrinkle" else class_num_list[dig]
            gt = self.labeling(label[name], class_num, bs).to(device)
            pred_p = softmax(pred[:, num : num + class_num])
            num += class_num
            loss += self.criterion(pred_p.type(torch.float), gt.type(torch.float))
            
            self.pred.append([dig, pred_p.argmax().item()])
            self.gt.append([dig, gt.argmax().item()])

        self.train_loss[self.m_idx].update(
            loss, batch_size=gt.shape[0]
        ) if self.phase == "train" else self.val_loss[self.m_idx].update(
            loss, batch_size=gt.shape[0]
        )

        return loss

    def regression(self, pred, label):
        loss = self.criterion(pred, label.to(device))
        self.train_loss[self.m_idx].update(
            loss, batch_size=pred.shape[0]
        ) if self.phase == "train" else self.val_loss[self.m_idx].update(
            loss, batch_size=pred.shape[0]
        )

        return loss


    def nan_detect(self, label):
        nan_list = list()
        for batch_data in label:
            for idx, data in enumerate(batch_data):
                if torch.isnan(data):
                    nan_list.append(idx)
        return nan_list

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

    def run(self, phase="train"):
        self.phase = phase
        self.model = (
            copy.deepcopy(self.model_list[self.m_idx])
            if self.phase == 'train'
            else self.temp_model_list[self.m_idx]
        )
        self.model.train() if self.phase == "train" else self.model.eval()
        
        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        adjust_learning_rate(optimizer, self.epoch, self.args)

        data_loader = (
            self.train_loader if self.phase == "train"
            else self.valid_loader
        )
        
        self.area_num = str(self.m_idx + 1) if self.flag else str(self.m_idx)
        self.img_count = 0
        
        def run_iter():
            self.img_count = 0
            for iteration, patch_list in enumerate(data_loader):
                if not self.area_num in list(patch_list.keys()):
                    continue

                if type(patch_list[self.area_num][1]) == torch.Tensor:
                    label = patch_list[self.area_num][1].to(device)
                    
                else:
                    for name in patch_list[self.area_num][1]:
                        patch_list[self.area_num][1][name] = patch_list[self.area_num][1][
                            name
                        ].to(device)
                    label = patch_list[self.area_num][1]

                if label == {}:
                    continue
                
                img = patch_list[self.area_num][0].to(device)
                if self.area_num in [4, 6]:
                    img = torch.flip(img, dims=[3])

                if img.shape[-1] > 128:
                    img_l = img[:, :, :, :128]
                    img_r = torch.flip(img[:, :, :, 128:], dims=[3])
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                elif img.shape[-2] > 128:
                    img_l = img[:, :, :128, :]
                    img_r = torch.flip(img[:, :, 128:, :], dims=[2])
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                else:
                    pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    loss = self.class_loss(pred, label)

                else:
                    idx_list = set([idx for idx in range(label.size(0))])
                    nan_list = set(self.nan_detect(label))
                    idx_list = list(idx_list - nan_list)
                    if len(idx_list) > 0:
                        loss = self.regression(pred[idx_list], label[idx_list])
                    else:
                        continue

                self.print_loss(iteration)

                if phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if iteration == len(data_loader) - 1:
                        self.temp_model_list[self.m_idx] = self.model
                        
                self.img_count += 1
                self.temp_model_list[self.m_idx] = self.model
        
        if phase == 'train':
            run_iter()
        else:
            with torch.no_grad():
                run_iter()
                