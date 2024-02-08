#!/usr/bin/env python3
import copy
import enum
import logging
import logging.handlers
import os
import shutil
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from src.configs.train_configs import LoggerConfig
from src.utils.metrics import Evaluator, metric_names


class Split(enum.Enum):
    VAL = "val"
    TRAIN = "train"
    TEST = "test"


def evaluator_add_batch(clip_output, logger):
    """Given the clip_output and logger, add the batch to the logger's evaluator."""
    logits = clip_output.logits_per_post.cpu().detach()
    pre_embeds = clip_output.post_embeds.cpu().detach()
    post_embeds = clip_output.post_embeds.cpu().detach()
    logger.evaluator.add_batch(logits, post_embeds, pre_embeds)


class Logger:
    def __init__(self, args, model=None):
        self.args = args
        self.args_save = copy.deepcopy(args)

        # Evaluator
        self.evaluator = Evaluator(self.args)

        # Checkpoint and Logging Directories
        self.dir_root = os.path.join(args.save_dir, args.name)
        self.dir_log = os.path.join(self.dir_root, "logs")
        self.dir_save = os.path.join(self.dir_root, "ckpts")

        if args.reset and os.path.exists(self.dir_root) and not args.finetune:
            shutil.rmtree(self.dir_root, ignore_errors=True)
        if not os.path.exists(self.dir_root):
            os.makedirs(self.dir_root)
        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)
        elif (
            os.path.exists(os.path.join(self.dir_save, "last.pth"))
            and os.path.exists(self.dir_log)
            and not args.finetune
        ):
            shutil.rmtree(self.dir_log, ignore_errors=True)
        if not os.path.exists(self.dir_log):
            os.makedirs(self.dir_log)

        if model is not None:
            self.model = model

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)

        # Log variables
        self.loss = 0
        self.auc = 0
        self.val_loss_mean = []
        self.best_auc = 0
        self.bestauc_iter = 0
        self.bestloss_iter = 0
        self.best_results = []
        self.best_loss = np.Inf

    def log_scalars(self, step):
        self.writer.add_scalar("loss", self.loss / self.log_iter, global_step=step)

    def loss_reset(self):
        self.loss = 0

    def add_logs(self, step, split: Split, loss):
        self.writer.add_scalar(f"{split.value}/loss", loss, global_step=step)

        if self.args.finetune or self.args.model == "ocp":
            if self.args.train_mode == "binary_class":
                f1, auc, apr, acc = self.evaluator.performance_task(validation=True)
                self.writer.add_scalar(f"{split.value}/f1", f1, global_step=step)
                self.writer.add_scalar(f"{split.value}/auc", auc, global_step=step)
                self.writer.add_scalar(f"{split.value}/apr", apr, global_step=step)
                self.writer.add_scalar(f"{split.value}/acc", acc, global_step=step)
            else:
                loss_value, r, pval = self.evaluator.performance_task(validation=True)
                self.writer.add_scalar(
                    f"{split.value}/loss_value", loss_value, global_step=step
                )
                self.writer.add_scalar(f"{split.value}/r", r, global_step=step)
                self.writer.add_scalar(f"{split.value}/pval", pval, global_step=step)

        else:
            (
                pre_auc,
                pre_accuracy,
                post_auc,
                post_accuracy,
                avg_logit,
                post_embed_avg,
                pre_embed_avg,
            ) = self.evaluator.performance_metric(validation=True)
            self.writer.add_scalar(f"{split.value}/pre_auc", pre_auc, global_step=step)
            self.writer.add_scalar(
                f"{split.value}/pre_accuracy", pre_accuracy, global_step=step
            )
            self.writer.add_scalar(
                f"{split.value}/post_auc", post_auc, global_step=step
            )
            self.writer.add_scalar(
                f"{split.value}/post_accuracy", post_accuracy, global_step=step
            )
            self.writer.add_scalar(
                f"{split.value}/avg_logit", avg_logit, global_step=step
            )
            self.writer.add_scalar(
                f"{split.value}/unnormalized_post_embed_abs_avg",
                post_embed_avg,
                global_step=step,
            )
            self.writer.add_scalar(
                f"{split.value}/unnormalized_pre_embed_abs_avg",
                pre_embed_avg,
                global_step=step,
            )

        if split == Split.VAL and self.best_loss > loss:
            self.best_loss = loss
            self.bestloss_iter = step

        self.writer.flush()

    def save(self, model, optimizer, step, last=None, finetune=None):
        self.model = model
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_results": self.best_results,
            "best_step": step,
            "last_step": last,
        }

        if step == self.bestauc_iter:
            self.save_ckpt(ckpt, "bestauc.pth")

        if step == self.bestloss_iter:
            self.save_ckpt(ckpt, "bestloss.pth")

        if last:
            self.save_ckpt(ckpt, "last.pth")
        self.save_ckpt(ckpt, "{}.pth".format(step))

        return ckpt

    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))


class TuneLogger:
    def __init__(self, args: LoggerConfig):
        self.args = args
        # Checkpoint and Logging Directories
        self.dir_root = args.save_dir
        assert os.path.exists(
            self.dir_root
        ), f"Save directory {self.dir_root} does not exist."
        self.dir_log = os.path.join(self.dir_root, "logs")
        os.makedirs(self.dir_log, exist_ok=True)
        self.dir_save = os.path.join(self.dir_root, "ckpts")
        os.makedirs(self.dir_save, exist_ok=True)

        # Tensorboard Writer
        self.writer = SummaryWriter(logdir=self.dir_log, flush_secs=60)

        self.batch_loss_step = dict()
        self.avg_loss_step = dict()
        self.best_loss = np.Inf
        self.bestloss_iter = 0

    def log_batch_loss(self, batch_loss, split):
        step = self.batch_loss_step.setdefault(split, 0)
        self.writer.add_scalar(
            f"{split.value}/batch_loss", batch_loss, global_step=step
        )
        self.batch_loss_step[split] = step + 1
        self.writer.flush()

    def log_avg_loss(self, avg_loss, split):
        step = self.avg_loss_step.setdefault(split, 0)
        self.writer.add_scalar(f"{split.value}/avg_loss", avg_loss, global_step=step)
        if split == Split.VAL and self.best_loss > avg_loss:
            self.best_loss = avg_loss
            self.bestloss_iter = step
        self.avg_loss_step[split] = step + 1
        self.writer.flush()

    def get_ckpt_dict(self, model, optimizer, step, last=None, finetune=None):
        self.model = model
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": self.best_loss,
            "best_step": self.bestloss_iter,
            "epoch": step,
            "last_step": last,
        }
        return ckpt

    def get_ckpt_folder(self, step):
        ckpt_folder = f"ckpt_{step}"
        os.makedirs(os.path.join(self.dir_save, ckpt_folder), exist_ok=True)
        return ckpt_folder

    def get_ckpt_directory(self, step):
        return os.path.join(self.dir_save, self.get_ckpt_folder(step))

    def save(self, model, optimizer, step, last=None, finetune=None, config=None):
        ckpt = self.get_ckpt_dict(model, optimizer, step, last, finetune)
        ckpt["config"] = config

        if step == self.bestloss_iter:
            self.save_ckpt(ckpt, "bestloss.pth")

        if last:
            self.save_ckpt(ckpt, "last.pth")

        ckpt_folder = self.get_ckpt_folder(step)
        self.save_ckpt(ckpt, f"{ckpt_folder}/ckpt.pth".format(step))

    def save_ckpt(self, ckpt, name):
        torch.save(ckpt, os.path.join(self.dir_save, name))
