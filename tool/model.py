from collections import defaultdict
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from scipy.stats import pearsonr
import glob
import shutil

import wandb

from tqdm import tqdm
from data_loader import mkdir
import torch.optim as optim
from utils import (
    AverageMeter,
    mape_loss,
    CB_loss,
    save_checkpoint,
    CharbonnierLoss,
    FocalLoss,
)
import os

from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Model(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.train_loss, self.val_loss = AverageMeter(), AverageMeter()
        self.keep_acc = {
            "sagging": AverageMeter(),
            "wrinkle_forehead": AverageMeter(),
            "wrinkle_glabellus": AverageMeter(),
            "wrinkle_perocular": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation_forehead": AverageMeter(),
            "pigmentation_cheek": AverageMeter(),
            "dryness": AverageMeter(),
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
        self.nan = 0
        (
            self.phase,
            self.update_c,
            self.stop_loss,
            self.device,
        ) = (None, 0, np.inf, device)
        self.pred, self.gt = list(), list()
        self.pred_t, self.gt_t = list(), list()

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        self.grad_accum_steps = max(1, int(getattr(self.args, "grad_accum_steps", 1)))

        total_epochs = max(1, int(self.args.epoch))
        warmup_epochs = min(
            max(0, int(getattr(self.args, "warmup_epochs", max(1, total_epochs // 10)))),
            max(0, total_epochs - 1),
        )

        base_lr = max(self.args.lr, 1e-8)
        lr_min = max(
            1e-8,
            float(getattr(self.args, "lr_min", base_lr * getattr(self.args, "lr_min_scale", 0.01))),
        )
        min_factor = min(1.0, max(lr_min / base_lr, 1e-6))

        scheduler_mode = getattr(self.args, "lr_scheduler", "cosine")
        milestones = sorted(set(int(m) for m in getattr(self.args, "decay_milestones", []) if m >= 0))
        decay_gamma = float(getattr(self.args, "decay_gamma", 0.5))
        decay_gamma = decay_gamma if decay_gamma > 0 else 0.1

        def lr_lambda(current_epoch: int):
            if warmup_epochs > 0 and current_epoch < warmup_epochs:
                return (current_epoch + 1) / float(max(1, warmup_epochs))

            if warmup_epochs >= total_epochs:
                return 1.0

            if scheduler_mode == "cosine":
                progress = (current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                progress = min(max(progress, 0.0), 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cosine * (1.0 - min_factor) + min_factor

            # multistep decay
            factor = 1.0
            for milestone in milestones:
                if current_epoch >= milestone:
                    factor *= decay_gamma
            return max(min_factor, factor)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self._norm_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def _denormalize_for_logging(self, tensor: torch.Tensor) -> torch.Tensor:
        """Undo dataset normalization for visualization."""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        img = tensor.detach()
        img = img * self._norm_std.to(img.device) + self._norm_mean.to(img.device)
        return img.clamp(0.0, 1.0).squeeze(0)

    def _handle_non_finite_loss(self, loss, pred, label):
        self.nan += 1
        loss_state = "nan" if torch.isnan(loss) else "inf"
        try:
            logits_min = pred.detach().min().item()
            logits_max = pred.detach().max().item()
        except Exception:
            logits_min, logits_max = float("nan"), float("nan")

        try:
            label_preview = (
                label.detach().cpu().tolist()
                if isinstance(label, torch.Tensor)
                else label
            )
        except Exception:
            label_preview = "unavailable"

        self.logger.warning(
            f"[{self.phase}] Non-finite loss detected ({loss_state}) "
            f"at epoch {self.epoch} iter {self.iter}. "
            f"logits range [{logits_min:.4f}, {logits_max:.4f}] labels {label_preview}"
        )

    def acc_avg(self, name):
        return round(self.test_value[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_value[name].avg, 4)

    def print_best(self):
        if self.args.mode == "class":
            self.logger.info(
                f"Best Epoch: {self.best_epoch}  Acc: {(self.acc_ * 100):.2f}%  Correlation: {self.corre_:.2f}"
            )

            for grade in sorted(self.correct_):
                print(
                    f"[Grade {grade}]  {self.correct_[grade]} / {self.all_[grade]} => {(self.correct_[grade] / self.all_[grade] * 100):.2f}%      ",
                    end="",
                )
            print("")

        else:
            self.logger.info(
                f"Best Epoch: {self.best_epoch}  MAE: {self.best_loss[self.m_dig]:.2f}  Correlation: {self.corre_:.2f}"
            )

        mkdir(
            os.path.join(
                "checkpoint",
                self.args.git_name,
                self.args.mode,
                self.args.name,
                "save_model",
                str(self.m_dig),
                "done",
            )
        )

    def print_loss(self, dataloader_len, final_flag=False):
        print(
            f"\r[{self.args.git_name}] Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}",
            end="",
        )
        loss_phase, loss_avg = (
            ("train_loss", self.train_loss.avg)
            if self.phase == "Train"
            else ("valid_loss", self.val_loss.avg)
        )
        self.wandb_run.log(
           {
               loss_phase: loss_avg,
               "lr": self.optimizer.param_groups[0]["lr"],
              "epoch": self.epoch,
               "global_step": self.global_step,
           },
           step=self.global_step,
        )

        if final_flag:
            f_pred = list()
            f_gt = list()

            correct_ = defaultdict(int)
            all_ = defaultdict(int)
            (pred_, gt_) = (
                (self.pred, self.gt)
                if self.phase == "Valid"
                else (self.pred_t, self.gt_t)
            )

            for value, value2 in zip(pred_, gt_):
                for v, v1 in zip(value, value2):
                    f_pred.append(v)
                    f_gt.append(v1)

            # safe Pearson: pearsonr returns nan for constant inputs
            try:
                if len(f_gt) > 1 and np.std(f_gt) > 0 and np.std(f_pred) > 0:
                    correlation, _ = pearsonr(f_gt, f_pred)
                else:
                    correlation = float("nan")
                    self.logger.warning(
                        f"Pearson correlation not defined (constant input) for {self.m_dig}: std(gt)={np.std(f_gt):.6f}, std(pred)={np.std(f_pred):.6f}, len={len(f_gt)}"
                    )
            except Exception as e:
                correlation = float("nan")
                self.logger.warning(f"Pearsonr failed for {self.m_dig}: {e}")

            if self.args.mode == "class":
                (
                    micro_precision,
                    _,
                    _,
                    _,
                ) = precision_recall_fscore_support(
                    f_gt, f_pred, average="micro", zero_division=1
                )

                for idx, i in enumerate(f_gt):
                    all_[i] += 1
                    if i == f_pred[idx]:
                        correct_[i] += 1

                info_m = (
                    f"[Lr: {self.optimizer.param_groups[0]['lr']:4f}][Gamma: {self.args.gamma}][Early Stop: {self.update_c}/{self.args.stop_early}]"
                    if self.phase == "Train"
                    else ""
                )
                self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}]{info_m}[{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}, Correlation: {correlation:.2f} micro Precision: {(micro_precision * 100):.2f}%"
                )

                if self.phase == "Valid":
                    if self.best_loss[self.m_dig] > self.val_loss.avg:
                        self.best_loss[self.m_dig] = round(self.val_loss.avg, 4)
                        save_checkpoint(
                            self, correct_, all_, micro_precision, correlation
                        )
                    else:
                        self.update_c += 1

            else:
                self.logger.info(
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}][Lr: {self.optimizer.param_groups[0]['lr']:4f}][Early Stop: {self.update_c}/{self.args.stop_early}][{self.m_dig}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}, Correlation: {correlation:.2f}"
                )
                if self.phase == "Valid":
                    if self.best_loss[self.m_dig] > self.val_loss.avg:
                        self.best_loss[self.m_dig] = round(self.val_loss.avg, 4)
                        save_checkpoint(self, None, None, None, correlation)
                    else:
                        self.update_c += 1

    def stop_early(self):
        if (self.update_c > self.args.stop_early) or (
            self.epoch == self.args.epoch - 1
        ):
            mkdir(
                os.path.join(
                    "checkpoint",
                    self.args.git_name,
                    self.args.mode,
                    self.args.name,
                    "save_model",
                    str(self.m_dig),
                    "done",
                )
            )
            return True

    def class_loss(self, pred, gt, loss=None):
        sample_loss = self.criterion(pred, gt) if loss is None else loss
        
        loss = sample_loss.mean()
        
        
        # if isinstance(self.criterion, nn.CrossEntropyLoss):
        #     if sample_loss.dim() == 0:
        #         loss = sample_loss
        #     else:
        #         distance_lambda = getattr(self.args, "distance_loss_weight", 0.0)
        #         if distance_lambda > 0:
        #             probs = torch.softmax(pred, dim=1)
        #             class_positions = torch.arange(
        #                 probs.size(1), device=probs.device, dtype=probs.dtype
        #             )
        #             expected_grade = (probs * class_positions).sum(dim=1)
        #             grade_distance = torch.abs(expected_grade - gt.to(probs.dtype))
        #             weight = 1.0 + distance_lambda * grade_distance
        #             sample_loss = sample_loss * weight
        #         loss = sample_loss.mean()

        # else:
        #     loss = sample_loss.mean()

        if not torch.isfinite(loss):
            self._handle_non_finite_loss(loss, pred, gt)
            return None

        with torch.no_grad():
            pred_v = [item.argmax().item() for item in pred]
            gt_v = [item.item() for item in gt]

            if self.phase == "Train":
                self.pred_t.append(pred_v)
                self.gt_t.append(gt_v)
                self.train_loss.update(loss.item(), batch_size=pred.shape[0])

            elif self.phase == "Valid":
                self.pred.append(pred_v)
                self.gt.append(gt_v)
                self.val_loss.update(loss.item(), batch_size=pred.shape[0])

        return loss

    def regression(self, pred, gt, loss=None):
        pred = pred.flatten()
        loss = self.criterion(pred.float(), gt.float()) if loss is None else loss

        if not torch.isfinite(loss):
            self._handle_non_finite_loss(loss, pred, gt)
            return None

        with torch.no_grad():
            pred_v = [item.item() for item in pred]
            gt_v = [item.item() for item in gt]

            if self.phase == "Train":
                self.pred_t.append(pred_v)
                self.gt_t.append(gt_v)
                self.train_loss.update(loss.item(), batch_size=pred.shape[0])

            elif self.phase == "Valid":
                self.pred.append(pred_v)
                self.gt.append(gt_v)
                self.val_loss.update(loss.item(), batch_size=pred.shape[0])

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

    def reset_log(self):
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.epoch += 1
        self.pred = list()
        self.gt = list()
        self.pred_t = list()
        self.gt_t = list()

    def update_e(self, epoch, **kwargs):
        self.epoch = self.best_epoch = epoch
        for key, value in kwargs.items():
            setattr(self, key, value)

    def train(self):
        self.model.train()
        self.phase = "Train"
        self.criterion = (
            # FocalLoss(gamma=self.args.gamma)
            # nn.CrossEntropyLoss()
            CB_loss(
                samples_per_cls=self.grade_num,
                no_of_classes=len(self.grade_num),
                gamma=self.args.gamma,
            )
            if self.args.mode == "class"
            else nn.HuberLoss()
        )
        self.prev_model = deepcopy(self.model)
        accum_steps = self.grad_accum_steps
        total_batches = len(self.train_loader)
        accum_counter = 0
        self.optimizer.zero_grad()

        for self.iter, (img, label, self.img_names, _, _, _) in enumerate(
            self.train_loader
        ):
            img, label = img.to(device), label.to(device)

            pred = self.model(img)

            if self.args.mode == "class":
                loss = self.class_loss(pred, label)
            else:
                loss = self.regression(pred, label)

            if loss is None:
                self.optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                continue

            if (
                self.wandb_run is not None
                and not self.iter
                and not self.epoch
                and img.size(0) > 0
            ):
                preview = []
                limit = min(3, img.size(0))
                for i in range(limit):
                    vis_img = self._denormalize_for_logging(img[i]).cpu()
                    caption = (
                        f"GT: {label[i].item()}, "
                        f"Pred: {pred[i].argmax().item()}, "
                        f"Name: {self.img_names[i]}"
                    )
                    preview.append(wandb.Image(vis_img, caption=caption))
                if preview:
                    self.wandb_run.log({"train/image": preview}, step=self.global_step)

            self.print_loss(len(self.train_loader))

            loss_to_backprop = loss / accum_steps
            loss_to_backprop.backward()
            accum_counter += 1

            should_step = (
                accum_counter == accum_steps or (self.iter + 1) == total_batches
            )
            if not should_step:
                continue

            if getattr(self.args, "grad_clip", 0) and self.args.grad_clip > 0:
                clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1
            accum_counter = 0

        self.print_loss(len(self.train_loader), final_flag=True)

    def valid(self):
        self.phase = "Valid"
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, self.img_names, _, _, _) in enumerate(
                self.valid_loader
            ):
                img, label = img.to(device), label.to(device)
                pred = self.model(img)

                if self.args.mode == "class":
                    loss = self.class_loss(pred, label)
                else:
                    loss = self.regression(pred, label)

                if loss is None:
                    continue

                if (
                    self.wandb_run is not None
                    and not self.iter
                    and not self.epoch
                    and img.size(0) > 0
                ):
                    preview = []
                    limit = min(3, img.size(0))
                    for i in range(limit):
                        vis_img = self._denormalize_for_logging(img[i]).cpu()
                        if self.args.mode == "class":
                            pred_value = pred[i].argmax().item()
                        else:
                            pred_value = round(pred[i].item(), 4)
                        caption = (
                            f"GT: {label[i].item()}, "
                            f"Pred: {pred_value}, "
                            f"Name: {self.img_names[i]}"
                        )
                        preview.append(wandb.Image(vis_img, caption=caption))
                    if preview:
                        self.wandb_run.log({"valid/image": preview}, step=self.global_step)

                self.print_loss(len(self.valid_loader))

            self.scheduler.step()
            self.print_loss(len(self.valid_loader), final_flag=True)


class Model_test(Model):
    def __init__(self, args, logger):
        self.args = args
        self.pred = defaultdict(lambda: defaultdict(list))
        self.gt = defaultdict(lambda: defaultdict(list))
        self.logger = logger
        self._save_path = None
        self._log_files_reset = False

    def _ensure_save_path(self):
        if self._save_path is None:
            self._save_path = os.path.join(
                self.args.log_path,
                str(self.args.equ),
                "save-log",
            )
        mkdir(self._save_path)
        self._reset_log_files(self._save_path)
        return self._save_path

    def _reset_log_files(self, save_path: str):
        """Prevent appending to old evaluation logs by clearing them once per run."""
        if self._log_files_reset:
            return

        if os.path.isdir(save_path):
            try:
                shutil.rmtree(save_path)
            except OSError as exc:
                self.logger.warning(f"Failed to clear old log directory {save_path}: {exc}")
        mkdir(save_path)

        for pattern in ("print_*.txt", "print_total.txt"):
            for file_path in glob.glob(os.path.join(save_path, pattern)):
                try:
                    os.remove(file_path)
                except OSError as exc:
                    self.logger.warning(f"Failed to remove old log file {file_path}: {exc}")
        self._log_files_reset = True

    def test(self, model, testset_loader, key):
        self.model = model
        self.testset_loader = testset_loader
        self.m_dig = key
        with torch.no_grad():
            self.model.eval()
            for self.iter, (img, label, self.img_names, self.digs, _, _) in enumerate(
                tqdm(self.testset_loader, desc=self.m_dig)
            ):
                img, label = img.to(device), label.to(device)
                pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    self.get_test_acc(pred, label)
                else:
                    self.get_test_loss(pred, label)

    def save_value(self):
        pred_path = os.path.join(
            self.args.check_root,
            "checkpoint",
            self.args.git_name,
            self.args.mode,
            self.args.name,
            "prediction",
            str(self.args.equ[0]),
        )
        mkdir(pred_path)

        with open(os.path.join(pred_path, f"pred.txt"), "w") as p:
            with open(os.path.join(pred_path, f"gt.txt"), "w") as g:
                for key in list(self.pred.keys()):
                    for angle in sorted(self.pred[key].keys()):
                        for p_v, g_v in zip(self.pred[key][angle], self.gt[key][angle]):
                            p.write(f"{angle}, {key}, {p_v[0]}, {p_v[1]} \n")
                            g.write(f"{angle}, {key}, {g_v[0]}, {g_v[1]} \n")
        g.close()
        p.close()

    def print_test(self):
        save_path = self._ensure_save_path()
        pred_total, gt_total = list(), list()
        for self.angle in sorted(self.pred[self.m_dig].keys()):
            gt_v = [value[0] for value in self.gt[self.m_dig][self.angle]]
            pred_v = [value[0] for value in self.pred[self.m_dig][self.angle]]

            pred_total.append(pred_v)
            gt_total.append(gt_v)
            self.print_maes(gt_v, pred_v, True)

        gt_v = [j for i in gt_total for j in i]
        pred_v = [j for i in pred_total for j in i]
        self.print_maes(gt_v, pred_v, False)

    def get_test_loss(self, pred, gt):
        if "elasticity_R2" in self.m_dig:
            value = 1

        elif "moisture" in self.m_dig:
            value = 100

        elif "wrinkle_Ra" in self.m_dig:
            value = 50

        elif self.m_dig == "pigmentation":
            value = 350

        elif "pore" in self.m_dig:
            value = 2600

        else:
            assert 0, "error"

        for idx, (pred_item, gt_item) in enumerate(zip(pred, gt)):
            self.pred[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [round(pred_item.item() * value, 3), self.img_names[idx], value]
            )
            self.gt[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [round(gt_item.item() * value, 3), self.img_names[idx], value]
            )

    def get_test_acc(self, pred, gt):
        for idx, (pred_item, gt_item) in enumerate(zip(pred, gt)):
            self.pred[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [pred_item.argmax().item(), self.img_names[idx]]
            )
            self.gt[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [gt_item.item(), self.img_names[idx]]
            )

    def print_maes(self, gt_v, pred_v, angle):
        correct_ = defaultdict(int)
        all_ = defaultdict(int)

        for gt, pred in zip(gt_v, pred_v):
            all_[gt] += 1
            if gt == pred:
                correct_[gt] += 1

        # safe Pearson: guard against constant arrays or other failures
        try:
            if len(gt_v) > 1 and np.std(gt_v) > 0 and np.std(pred_v) > 0:
                correlation, p_value = pearsonr(gt_v, pred_v)
            else:
                correlation = float("nan")
                p_value = float("nan")
                self.logger.warning(
                    f"Pearson correlation not defined (constant input) for {self.m_dig} angle={self.angle if hasattr(self, 'angle') else 'N/A'}: std(gt)={np.std(gt_v):.6f}, std(pred)={np.std(pred_v):.6f}, len={len(gt_v)}"
                )
        except Exception as e:
            correlation = float("nan")
            p_value = float("nan")
            self.logger.warning(f"Pearsonr failed for {self.m_dig}: {e}")
        save_path = self._ensure_save_path()

        if self.args.mode == "regression":
            n_gt_v = [value / max(gt_v) for value in gt_v]
            n_pred_v = [value / max(pred_v) for value in pred_v]

            mae = mean_absolute_error(gt_v, pred_v)
            mape = mape_loss()(np.array(pred_v), np.array(gt_v))
            nmae = mean_absolute_error(n_gt_v, n_pred_v)

            if angle:
                self.logger.info(
                    f"[{self.angle}][{self.m_dig}]Correlation: {correlation:.2f}, P-value: {p_value:.4f}, MAE: {mae:.4f}, MAPE: {mape:.3f}, NMAE: {nmae:.3f}"
                )

                file_path = os.path.join(save_path, f"print_{self.angle}.txt")
                file_exists = os.path.exists(file_path)
                with open(file_path, "a") as f:
                    if self.m_dig == "pigmentation" and not file_exists:
                        f.write(f"Angle, Area, Correlation, P-value, MAE, MAPE, NMAE\n")
                    f.write(
                        f"{self.angle}, {self.m_dig}, {correlation:.2f}, {p_value:.4f}, {mae:.2f}, {mape:.2f}, {nmae:.2f}\n"
                    )

            else:
                file_path = os.path.join(save_path, "print_total.txt")
                file_exists = os.path.exists(file_path)
                with open(file_path, "a") as f:
                    if self.m_dig == "pigmentation" and not file_exists:
                        f.write(f"Area, Correlation, P-value, MAE, MAPE, NMAE\n")
                    f.write(
                        f"{self.m_dig}, {correlation:.2f}, {p_value:.4f}, {mae:.2f}, {mape:.2f}, {nmae:.2f}\n"
                    )

        else:
            mae_ = [abs(p - g) for p, g in zip(pred_v, gt_v)]
            mae_ = sum(mae_) / len(mae_)

            mae_0 = [True if abs(p - g) == 0 else False for p, g in zip(pred_v, gt_v)]
            mae_0 = sum(mae_0) / len(mae_0)

            mae_1 = [True if abs(p - g) <= 1 else False for p, g in zip(pred_v, gt_v)]
            mae_1 = sum(mae_1) / len(mae_1)

            mae_2 = [True if abs(p - g) <= 2 else False for p, g in zip(pred_v, gt_v)]
            mae_2 = sum(mae_2) / len(mae_2)

            if angle:
                self.logger.info(
                    f"[{self.angle}][{self.m_dig}]Correlation: {correlation:.2f}, P-value: {p_value:.4f}, MAE: {mae_:.2f}, MAE(==0): {mae_0 * 100:.2f}%,  MAE(=<1): {mae_1 * 100:.2f}%, MAE(=<2): {mae_2 * 100:.2f}%"
                )
                for grade in all_:
                    self.logger.info(
                        f"          {grade} grade Acc: {correct_[grade]} / {all_[grade]} -> {(correct_[grade]/all_[grade] * 100):.2f} %"
                    )
                file_path = os.path.join(save_path, f"print_{self.angle}.txt")
                file_exists = os.path.exists(file_path)
                with open(file_path, "a") as f:
                    if self.m_dig == "dryness" and not file_exists:
                        f.write(
                            f"Angle, Area, Correlation, P-value, MAE, MAE(==0), MAE(=<1), MAE(=<2)\n"
                        )
                    f.write(
                        f"{self.angle}, {self.m_dig}, {correlation:.2f}, {p_value:.4f}, {mae_:.2f}, {mae_0 * 100:.2f}, {mae_1 * 100:.2f}, {mae_2 * 100:.2f}\n"
                    )

            else:
                file_path = os.path.join(save_path, "print_total.txt")
                file_exists = os.path.exists(file_path)
                with open(file_path, "a") as f:
                    if self.m_dig == "dryness" and not file_exists:
                        f.write(
                            f"Area, Correlation, P-value, MAE, MAE(==0), MAE(=<1), MAE(=<2)\n"
                        )
                    f.write(
                        f"{self.m_dig}, {correlation:.2f}, {p_value:.4f}, {mae_:.2f}, {mae_0 * 100:.2f}, {mae_1 * 100:.2f}, {mae_2 * 100:.2f}\n"
                    )
