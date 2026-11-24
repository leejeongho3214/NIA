from collections import defaultdict
from copy import deepcopy
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import numpy as np
from scipy.stats import pearsonr
import glob

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

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


def paper_metrics(gt_v, pred_v, mode):
    """Metrics as reported in the paper.

    class: pred_v is the (continuous) predicted grade. MAE and Pearson r are computed on it;
           Exact / <=1 / <=2 accuracy on rint(pred) clipped to the GT range of the scored rows.
    regression: MAE, MAPE, Pearson r and NMAE = MAE / (max GT - min GT).
    """
    gt = np.asarray(gt_v, dtype=float)
    pred = np.asarray(pred_v, dtype=float)
    out = {"n": int(gt.size)}
    out["mae"] = float(np.mean(np.abs(pred - gt))) if gt.size else float("nan")
    if gt.size > 1 and np.std(gt) > 0 and np.std(pred) > 0:
        out["r"], out["p"] = (float(v) for v in pearsonr(gt, pred))
    else:
        out["r"], out["p"] = float("nan"), float("nan")
    gt_range = float(gt.max() - gt.min()) if gt.size else float("nan")
    if mode == "class":
        rounded = np.clip(np.rint(pred), gt.min(), gt.max())
        out["acc0"] = float(np.mean(rounded == gt) * 100)
        out["acc1"] = float(np.mean(np.abs(rounded - gt) <= 1) * 100)
        out["acc2"] = float(np.mean(np.abs(rounded - gt) <= 2) * 100)
    else:
        out["nmae"] = out["mae"] / gt_range if gt_range > 0 else float("nan")
        with np.errstate(divide="ignore", invalid="ignore"):
            out["mape"] = float(np.mean(np.abs(pred - gt) / gt))
    return out


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
        # expected grade (sum_k softmax_k * k) per sample, class mode only
        self.pred_corr, self.gt_corr = list(), list()
        self.pred_corr_t, self.gt_corr_t = list(), list()
        self.best_score = {key: float("inf") for key in getattr(self, "model_num_class", {})}

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
            ordinal_mae = getattr(self, "ordinal_mae_", float("nan"))
            self.logger.info(
                f"Best Epoch: {self.best_epoch}  Acc: {(self.acc_ * 100):.2f}%  "
                f"Correlation: {self.corre_:.2f}  Ordinal MAE: {ordinal_mae:.4f}"
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
                self.args.mode,
                self.args.name,
                "save_model",
                str(self.m_dig),
                "done",
            )
        )

    def print_loss(self, dataloader_len, final_flag=False):
        print(
            f"\r Epoch: {self.epoch} [{self.phase}][{self.m_dig}][{self.iter}/{dataloader_len}] ---- >  loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}",
            end="",
        )
        loss_phase, loss_avg = (
            ("train_loss", self.train_loss.avg)
            if self.phase == "Train"
            else ("valid_loss", self.val_loss.avg)
        )
        if self.wandb_run is not None:
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
            if self.phase == "Valid":
                pred_, gt_ = self.pred, self.gt
                pred_corr_, gt_corr_ = self.pred_corr, self.gt_corr
            else:
                pred_, gt_ = self.pred_t, self.gt_t
                pred_corr_, gt_corr_ = self.pred_corr_t, self.gt_corr_t

            for value, value2 in zip(pred_, gt_):
                for v, v1 in zip(value, value2):
                    f_pred.append(v)
                    f_gt.append(v1)

            f_pred_corr = [v for value in pred_corr_ for v in value]
            f_gt_corr = [v for value in gt_corr_ for v in value]

            # class mode: Pearson on the expected grade (paper metric); regression: raw output
            corr_pred = f_pred_corr if self.args.mode == "class" else f_pred
            corr_gt = f_gt_corr if self.args.mode == "class" else f_gt

            # safe Pearson: pearsonr returns nan for constant inputs
            try:
                if len(corr_gt) > 1 and np.std(corr_gt) > 0 and np.std(corr_pred) > 0:
                    correlation, _ = pearsonr(corr_gt, corr_pred)
                else:
                    correlation = float("nan")
                    self.logger.warning(
                        f"Pearson correlation not defined (constant input) for {self.m_dig}: std(gt)={np.std(corr_gt):.6f}, std(pred)={np.std(corr_pred):.6f}, len={len(corr_gt)}"
                    )
            except Exception as e:
                correlation = float("nan")
                self.logger.warning(f"Pearsonr failed for {self.m_dig}: {e}")

            if self.args.mode == "class":
                ordinal_mae = (
                    mean_absolute_error(f_gt_corr, f_pred_corr)
                    if f_gt_corr and f_pred_corr
                    else float("nan")
                )
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
                    f"Epoch: {self.epoch} [{self.phase}][{self.m_dig}]{info_m}[{self.iter}/{dataloader_len}] ---- >  "
                    f"loss: {self.train_loss.avg if self.phase == 'Train' else self.val_loss.avg:.04f}, "
                    f"Ordinal MAE: {ordinal_mae:.4f}, Correlation: {correlation:.2f} "
                    f"micro Precision: {(micro_precision * 100):.2f}%"
                )

                if self.phase == "Valid":
                    # model selection on the validation split (default: MAE of the expected grade)
                    checkpoint_metric = getattr(self.args, "checkpoint_metric", "ordinal_mae")
                    if checkpoint_metric == "corr" and np.isfinite(correlation):
                        current_score = -correlation
                    elif checkpoint_metric == "ordinal_mae" and np.isfinite(ordinal_mae):
                        current_score = ordinal_mae
                    else:
                        current_score = self.val_loss.avg
                    if self.best_score.get(self.m_dig, float("inf")) > current_score:
                        self.best_score[self.m_dig] = current_score
                        self.best_loss[self.m_dig] = round(self.val_loss.avg, 4)
                        self.ordinal_mae_ = ordinal_mae
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
        # Called after reset_log(), so self.epoch == number of finished epochs.
        # Stop after `stop_early` consecutive validation checks without improvement,
        # or once all `epoch` epochs have been trained.
        if (self.update_c >= self.args.stop_early) or (
            self.epoch >= self.args.epoch
        ):
            mkdir(
                os.path.join(
                    "checkpoint",
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

        prob = torch.softmax(pred, dim=1)
        grades = torch.arange(pred.shape[1], device=pred.device, dtype=prob.dtype)
        expected = (prob * grades.unsqueeze(0)).sum(dim=1)

        # ordinal term: smooth-L1 between the expected grade and the GT grade
        ordinal_weight = float(getattr(self.args, "ordinal_loss_weight", 0.0))
        if ordinal_weight > 0:
            loss = loss + ordinal_weight * nn.functional.smooth_l1_loss(expected, gt.float())
        
        
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
            pred_corr_v = [item.item() for item in expected]
            gt_corr_v = [float(item.item()) for item in gt]

            if self.phase == "Train":
                self.pred_t.append(pred_v)
                self.gt_t.append(gt_v)
                self.pred_corr_t.append(pred_corr_v)
                self.gt_corr_t.append(gt_corr_v)
                self.train_loss.update(loss.item(), batch_size=pred.shape[0])

            elif self.phase == "Valid":
                self.pred.append(pred_v)
                self.gt.append(gt_v)
                self.pred_corr.append(pred_corr_v)
                self.gt_corr.append(gt_corr_v)
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
        self.pred_corr = list()
        self.gt_corr = list()
        self.pred_corr_t = list()
        self.gt_corr_t = list()

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
            self._save_path = os.path.join(self.args.log_path, "save-log")
        mkdir(self._save_path)
        self._reset_log_files(self._save_path)
        return self._save_path

    def _reset_log_files(self, save_path: str):
        """Remove this tool's own old print_*.txt files once per run (never whole directories)."""
        if self._log_files_reset:
            return
        for file_path in glob.glob(os.path.join(save_path, "print_*.txt")):
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

    def _scored_pred(self, p_v):
        """Prediction used for scoring: expected grade (paper) or argmax (--argmax) in class mode."""
        if self.args.mode == "class" and not getattr(self.args, "argmax", False):
            return p_v[2]
        return p_v[0]

    def save_value(self):
        pred_path = os.path.join(
            self.args.check_root,
            "checkpoint",
            self.args.mode,
            self.args.name,
            "prediction",
        )
        mkdir(pred_path)

        with open(os.path.join(pred_path, "pred.txt"), "w") as p, open(
            os.path.join(pred_path, "gt.txt"), "w"
        ) as g:
            for key in list(self.pred.keys()):
                for angle in sorted(self.pred[key].keys()):
                    for p_v, g_v in zip(self.pred[key][angle], self.gt[key][angle]):
                        p.write(f"{angle}, {key}, {self._scored_pred(p_v)}, {p_v[1]} \n")
                        g.write(f"{angle}, {key}, {g_v[0]}, {g_v[1]} \n")

    def print_pooled(self):
        """Paper protocol (Table 6): one score over all test images of all tasks for this seed."""
        gt_v = [g[0] for key in self.gt for angle in self.gt[key] for g in self.gt[key][angle]]
        pred_v = [
            self._scored_pred(p)
            for key in self.pred
            for angle in self.pred[key]
            for p in self.pred[key][angle]
        ]
        m = paper_metrics(gt_v, pred_v, self.args.mode)
        if self.args.mode == "class":
            msg = (
                f"[POOLED over tasks] n={m['n']}  MAE: {m['mae']:.4f}  Pearson r: {m['r']:.4f}  "
                f"MAE=0: {m['acc0']:.2f}%  MAE<=1: {m['acc1']:.2f}%  MAE<=2: {m['acc2']:.2f}%"
            )
        else:
            msg = (
                f"[POOLED over tasks] n={m['n']}  (regression targets have different units; "
                f"see per-task NMAE)  MAE: {m['mae']:.4f}  Pearson r: {m['r']:.4f}"
            )
        self.logger.info(msg)
        with open(os.path.join(self._ensure_save_path(), "print_pooled.txt"), "w") as f:
            f.write(", ".join(m.keys()) + "\n")
            f.write(", ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in m.values()) + "\n")
        return m

    def print_test(self):
        pred_total, gt_total = list(), list()
        for self.angle in sorted(self.pred[self.m_dig].keys()):
            gt_v = [value[0] for value in self.gt[self.m_dig][self.angle]]
            pred_v = [self._scored_pred(value) for value in self.pred[self.m_dig][self.angle]]

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
            prob = torch.softmax(pred_item, dim=0)
            grades = torch.arange(pred_item.numel(), device=pred_item.device, dtype=prob.dtype)
            expected = (prob * grades).sum().item()
            self.pred[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [pred_item.argmax().item(), self.img_names[idx], expected]
            )
            self.gt[self.m_dig][self.img_names[idx].split("_")[-3]].append(
                [gt_item.item(), self.img_names[idx]]
            )

    def print_maes(self, gt_v, pred_v, angle):
        m = paper_metrics(gt_v, pred_v, self.args.mode)
        save_path = self._ensure_save_path()
        where = f"[{self.angle}][{self.m_dig}]" if angle else f"[{self.m_dig}]"
        file_name = f"print_{self.angle}.txt" if angle else "print_total.txt"
        file_path = os.path.join(save_path, file_name)
        file_exists = os.path.exists(file_path)

        if self.args.mode == "regression":
            self.logger.info(
                f"{where}Correlation: {m['r']:.2f}, P-value: {m['p']:.4f}, MAE: {m['mae']:.4f}, "
                f"MAPE: {m['mape']:.3f}, NMAE: {m['nmae']:.3f}"
            )
            with open(file_path, "a") as f:
                if not file_exists:
                    f.write(("Angle, " if angle else "") + "Area, Correlation, P-value, MAE, MAPE, NMAE\n")
                f.write(
                    (f"{self.angle}, " if angle else "")
                    + f"{self.m_dig}, {m['r']:.2f}, {m['p']:.4f}, {m['mae']:.2f}, {m['mape']:.2f}, {m['nmae']:.3f}\n"
                )
        else:
            self.logger.info(
                f"{where}Correlation: {m['r']:.2f}, P-value: {m['p']:.4f}, MAE: {m['mae']:.2f}, "
                f"MAE(==0): {m['acc0']:.2f}%,  MAE(=<1): {m['acc1']:.2f}%, MAE(=<2): {m['acc2']:.2f}%"
            )
            with open(file_path, "a") as f:
                if not file_exists:
                    f.write(
                        ("Angle, " if angle else "")
                        + "Area, Correlation, P-value, MAE, MAE(==0), MAE(=<1), MAE(=<2)\n"
                    )
                f.write(
                    (f"{self.angle}, " if angle else "")
                    + f"{self.m_dig}, {m['r']:.2f}, {m['p']:.4f}, {m['mae']:.2f}, "
                    f"{m['acc0']:.2f}, {m['acc1']:.2f}, {m['acc2']:.2f}\n"
                )
