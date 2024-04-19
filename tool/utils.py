import random
import cv2
import numpy as np
import torch
import inspect
import errno
import os
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    return e_x / torch.sum(e_x, dim=1).unsqueeze(dim=1)


def resume_checkpoint(args, model, path):
    state_dict = torch.load(path, map_location=device)
    args.best_loss = state_dict["best_loss"]
    args.load_epoch = state_dict["epoch"]
    if 'batch_size' in state_dict:
        args.batch_size = state_dict["batch_size"]
        if args.batch_size != state_dict['batch_size']:
            print(f"batch_size update 128 ->> {args.batch_size}")
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict

    return model

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target, dig):
        gt = target.item()
        
        smoothing = 0.5
        dim = 0
        if (dig == 'wrinkle' and gt == 1) or (dig == 'pigmentation' and gt ==1):
            target = torch.tensor([0, 1, 2], device = "cuda")
            
        elif dig == 'sagging' and gt == 0:
            target = torch.tensor([0, 1], device = "cuda")
            
        elif (dig == 'pore' and gt ==2) or (dig == 'dryness' and gt ==2):
            target = torch.tensor([1, 2, 3], device = "cuda")
            
        else:
            smoothing = self.smoothing
            dim = 1

        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(dim))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()


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


def get_item(item, device):
    if type(item[1]) == torch.Tensor:
        label = item[1].to(device)

    else:
        for name in item[1]:
            item[1][name] = item[1][name].to(device)
        label = item[1]

    img = item[0].to(device)

    return img, label


def pred_image(self, img):
    if img.shape[-1] > 128:
        img_l = img[:, :, :, :128]
        img_r = torch.flip(img[:, :, :, 128:], dims=[3])
        pred = self.model.to(self.device)(img_l)
        pred = self.model.to(self.device)(img_r) + pred

    elif img.shape[-2] > 128:
        img_l = img[:, :, :128, :]
        img_r = torch.flip(img[:, :, 128:, :], dims=[2])
        pred = self.model.to(self.device)(img_l)
        pred = self.model.to(self.device)(img_r) + pred

    else:
        pred = self.model.to(self.device)(img)

    return pred

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, epoch = 0, alpha=1, gamma=3, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma 
        self.reduction = reduction
        
        if gamma != 3 and epoch == 0:
            print("현재 감마 3 아니다~")

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def save_checkpoint(self):
    checkpoint_dir = os.path.join(self.args.output_dir, self.args.mode, self.args.name, "save_model", str(self.m_dig))
    mkdir(checkpoint_dir)
    model_to_save = self.model.module if hasattr(self.model, "module") else self.model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "batch_size": self.args.batch_size
        },
        os.path.join(checkpoint_dir, "temp_file.bin"),
    )

    os.rename(
        os.path.join(checkpoint_dir, "temp_file.bin"),
        os.path.join(checkpoint_dir, "state_dict.bin"),
    )
    
    return checkpoint_dir


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


def save_value(args, self):
    mkdir(args.name)
    with open(os.path.join(args.name, f"pred.txt"), "w") as p:
        with open(os.path.join(args.name, f"gt.txt"), "w") as g:
            for idx in range(len(self.pred)):
                p.write(f"{self.pred[idx][0]}, {self.pred[idx][1]} \n")
                g.write(f"{self.gt[idx][0]}, {self.gt[idx][1]} \n")
    g.close()
    p.close()


def save_image(self, img):
    c_img = make_grid(img, nrow = int(self.args.batch_size / 4)).permute(1, 2, 0).detach().cpu().numpy() 
    max_v, min_v = c_img.max(), c_img.min()
    if min_v > 0:
        min_v = -min_v
    s_img = (c_img - min_v) * (255.0 / (max_v - min_v))

    path = os.path.join(self.args.save_img, self.m_dig)
    mkdir(path)
    img_mat = cv2.UMat(s_img)
    j = self.args.batch_size // 4
    for i, name in enumerate(self.img_names):
        x, y = (i % j * (256 + 2) + 2, i // j * (256 + 2) + 20)  # 위치 조절
        cv2.putText(img_mat, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(path, f"epoch_{self.epoch}_iter_{self.iter}_{self.m_dig}.jpg"), img_mat.get())

def fix_seed(random_seed):

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
