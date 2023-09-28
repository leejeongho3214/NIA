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
    state_dict = torch.load(path)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict
    args.load_epoch = epoch

    args.best_loss = best_loss

    return model


class Model_test(object):
    def __init__(self, args, model_list, testset_loader):
        super(Model_test, self).__init__()
        self.args = args
        self.model_list = model_list

        self.test_loader = testset_loader

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

        self.equip_loss = {
            "0": {"count": 1},
            "1": {"moisture": 1, "elasticity": 1},
            "3": {"wrinkle": 1},
            "4": {"wrinkle": 1},
            "5": {"moisture": 1, "elasticity": 1, "pore": 1},
            "6": {"moisture": 1, "elasticity": 1, "pore": 1},
            "8": {"moisture": 1, "elasticity": 1},
        }
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0
        self.logger = None

    def choice(self, m_idx):
        if m_idx in [4, 6]:
            m_idx -= 1
            self.flag = True
        else:
            self.flag = False

        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def acc_avg(self, name):
        return round(self.test_class_acc[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_regresion_mae[name].avg, 4)

    def print_total(self):
        if self.args.mode == "class":
            print(
                f"[{self.phase}] pigmentation: {self.acc_avg('pigmentation')}% // wrinkle: {self.acc_avg('wrinkle')}% // sagging: {self.acc_avg('sagging')}% // pore: {self.acc_avg('pore')}% // dryness: {self.acc_avg('dryness')}%"
            )

        else:
            print(
                f"[{self.phase}] moisture: {self.loss_avg('moisture')} // wrinkle: {self.loss_avg('wrinkle')} // elasticity: {self.loss_avg('elasticity')} // pore: {self.loss_avg('pore')}"
            )

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

    def get_test_acc(self, pred, label, patch_list, area_num):
        gt = (
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
            .permute(1, 0)
            .to(device)
        )
        num = 0
        patch_list[area_num].append(dict())
        for idx, area_name in enumerate(label):
            dig = area_name.split("_")[-1]
            pred_l = torch.argmax(pred[:, num : num + class_num_list[dig]], dim=1)
            num += class_num_list[dig]
            self.test_class_acc[dig].update_acc(
                (pred_l == gt[:, idx]).sum().item(),
                pred_l.shape[0],
            )
            patch_list[area_num][2][dig] = [int(gt[:, idx].item()), int(pred_l.item())]

        return patch_list

    def save_img(self, patch_list, iteration):
        self.num_patch = len(patch_list) + 5
        self.col = 4 if self.args.mode == 'class' else 3
        vis_img = np.zeros([256 * 4, 256 * self.col, 3])


        self.num = 0
        for area_num in patch_list:
            img = patch_list[area_num][0][0].permute(1, 2, 0)
            if self.args.normalize:
                img = (img + 1) / 2

            if int(area_num) in [1, 7, 8]:
                l_img = (img[:, :128]).numpy()
                l_img = cv2.resize(l_img, (256, 256))
                cv2.putText(
                    l_img,
                    f"{area_naming[str(int(area_num))]}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 244, 0),
                    2,
                )
                for idx, name in enumerate(patch_list[area_num][2]):
                    cv2.putText(
                        l_img,
                        f"Gt {name} => {patch_list[area_num][2][name][0]}",
                        (0, l_img.shape[0] - 50 - 50 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 244, 244),
                        2,
                    )
                    cv2.putText(
                        l_img,
                        f"Pred {name} => {patch_list[area_num][2][name][1]}",
                        (0, l_img.shape[0] - 25 - 50 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (100, 244, 244),
                        2,
                    )
                vis_img = self.match_img(vis_img, l_img)
                self.num += 1
                r_img = (img[:, 128:]).numpy()
                r_img = cv2.resize(r_img, (256, 256))
                cv2.putText(
                    r_img,
                    f"{area_naming[str(int(area_num))]}",
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
                    f"{area_naming[str(int(area_num))]}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 244, 0),
                    2,
                )
                for idx, name in enumerate(patch_list[area_num][2]):
                    cv2.putText(
                        l_img,
                        f"Gt {name} => {patch_list[area_num][2][name][0]}",
                        (0, l_img.shape[0] - 50 - 50 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 244, 244),
                        2,
                    )
                    cv2.putText(
                        l_img,
                        f"Pred {name} => {patch_list[area_num][2][name][1]}",
                        (0, l_img.shape[0] - 25 - 50 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (100, 244, 244),
                        2,
                    )
                vis_img = self.match_img(vis_img, l_img)
                self.num += 1
                r_img = (img[128:]).numpy()
                r_img = cv2.resize(r_img, (256, 256))
                cv2.putText(
                    r_img,
                    f"{area_naming[str(int(area_num))]}",
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
                    f"{area_naming[str(int(area_num))]}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 244, 0),
                    2,
                )
                for idx, name in enumerate(patch_list[area_num][2]):
                    cv2.putText(
                        img,
                        f"Gt {name} => {patch_list[area_num][2][name][0]}",
                        (0, img.shape[0] - 50 - 50 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 244, 244),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Pred {name} => {patch_list[area_num][2][name][1]}",
                        (0, img.shape[0] - 25 - 50 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (100, 244, 244),
                        2,
                    )
                vis_img = self.match_img(vis_img, img)
                self.num += 1

        mkdir(f"vis/test/{self.args.mode}/{self.args.name}")
        cv2.imwrite(
            f"vis/test/{self.args.mode}/{self.args.name}/test_{iteration}.jpg",
            vis_img * 255,
        )

    def get_test_loss(self, pred_p, label, area_num, patch_list):
        count = 0

        patch_list[area_num].append(dict())
        for name in self.equip_loss[area_num]:
            dig = name.split("_")[-1]
            gt = label[:, count : count + self.equip_loss[area_num][name]]
            pred = pred_p[:, count : count + self.equip_loss[area_num][name]]
            count += self.equip_loss[area_num][name]
            self.test_regresion_mae[name].update(
                self.criterion(pred, gt).item(), batch_size=pred.shape[0]
            )
            if dig == 'moisture':
                gt, pred = gt * 100, pred * 100
            elif dig == 'count':
                gt, pred = gt * 300, pred * 300
            elif dig == 'pore':
                gt, pred = gt * 3000, pred * 3000
            patch_list[area_num][2][dig] = [round(gt.item(), 3), round(pred.item(), 3)]

        return patch_list

    def test(self, model_num_class, data_loader):
        for iteration, patch_list in enumerate(data_loader):
            for model_idx in range(len(model_num_class)):
                if np.isnan(model_num_class[model_idx]):
                    continue

                self.choice(model_idx)
                self.model = self.model_list[self.m_idx]
                self.model.eval()

                data_loader = self.test_loader
                area_num = str(self.m_idx + 1) if self.flag else str(self.m_idx)

                if type(patch_list[area_num][1]) == torch.Tensor:
                    label = patch_list[area_num][1].to(device)
                else:
                    for name in patch_list[area_num][1]:
                        patch_list[area_num][1][name] = patch_list[area_num][1][
                            name
                        ].to(device)
                    label = patch_list[area_num][1]

                img = patch_list[area_num][0].to(device)

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
                    patch_list = self.get_test_acc(pred, label, patch_list, area_num)
                else:
                    patch_list = self.get_test_loss(
                        pred, label.to(device), area_num, patch_list
                    )

            self.save_img(patch_list, iteration)
