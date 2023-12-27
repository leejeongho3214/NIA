import cv2
import torch
import copy
import errno
import os
import torch.nn as nn
import torch.nn.functional as F
from data_loader import area_naming

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
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

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



def labeling(gt, num):
    template = torch.zeros(num)
    label = gt.item()
    
    if label == 0:
        template[0] = 0.85
        template[1] = 0.1

    elif label == num - 1:
        template[-1] = 0.85
        template[-2] = 0.1

    else:
        template[label] = 0.85
        template[label - 1] = 0.05
        template[label + 1] = 0.05
    
    zero_index = torch.where(template == 0)[0]
    template[zero_index] = 0.05 / len(zero_index)

    return template.reshape(1, -1).cuda()


def save_checkpoint(model, args, epoch, m_dig, best_loss):
    checkpoint_dir = os.path.join(args.output_dir, args.mode, args.name, str(m_dig))
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
    c_img =img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    max_v, min_v = c_img.max(), c_img.min()
    if min_v > 0:
        min_v = -min_v
    s_img = (c_img - min_v) * (255.0 / (max_v - min_v))

    path = os.path.join(self.args.save_img, self.m_dig)
    mkdir(path)
    cv2.imwrite(os.path.join(path, f"epoch_{self.epoch}_{self.m_dig}.jpg"), s_img)

def img_save(self, img, label):
    c_img =img.detach().cpu().numpy()[0].transpose(1, 2, 0)
    max_v, min_v = c_img.max(), c_img.min()
    if min_v > 0:
        min_v = -min_v
    s_img = (c_img - min_v) * (255.0 / (max_v - min_v))

    path = os.path.join(self.args.save_img, self.m_dig)
    mkdir(path)
    cv2.imwrite(os.path.join(path, f"[{self.iter}]_{area_naming[str(int(self.area))]}_{label.item()}.jpg"), s_img)