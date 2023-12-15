import cv2
import torch
import copy
import errno
import os
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self, x, target, smoothing=0.1):
        confidence = 1.0 - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
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


def labeling(label, num):
    # template = torch.zeros(num)
    # gt = label.item()
    # if gt == 0:
    #     template[0] = 0.6
    #     template[1] = 0.3
    #     template[2] = 0.1

    # elif gt == num - 1:
    #     template[-1] = 0.6
    #     template[-2] = 0.3
    #     template[-3] = 0.1

    # else:
    #     template[gt] = 0.5
    #     template[gt - 1] = 0.25
    #     template[gt + 1] = 0.25

    # return template.reshape(1, -1)

    return F.one_hot(label, num)


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


def save_image(self, patch_list, epoch):
    for dig, patches in patch_list.items():
        img_list = list()
        for patch in patches:
            img_list.append(patch[0])
        
        try:
            img = torch.cat(img_list, dim=3)
        except:
            try:
                img = torch.cat(img_list, dim=2)
            except:
                continue
            
        c_img = img.detach().cpu().numpy()[0].transpose(1, 2, 0)
        max_v, min_v = c_img.max(), c_img.min()
        if min_v > 0:
            min_v = -min_v
        s_img = (c_img - min_v) * (255.0 / (max_v - min_v))

        path = os.path.join("save_img", self.args.mode, self.args.name, dig)
        mkdir(path)
        cv2.imwrite(os.path.join(path, f"epoch_{epoch}.jpg"), s_img)

    return
