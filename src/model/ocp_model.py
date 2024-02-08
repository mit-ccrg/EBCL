import dataclasses

import torch
from torch import nn

from typing import Union

import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl

from src.model.utils import OutputBase
from src.configs.model_configs import StratsModelConfig
from src.configs.dataloader_configs import SupervisedView
from src.model.supervised_model import SupervisedModule

from .architectures.triplet_transformer import TransformerModel


@dataclasses.dataclass
class OCPPretrainOutput(OutputBase):
    loss: torch.Tensor
    logits: torch.Tensor
    attn: torch.Tensor
    outputs: torch.Tensor


class OCPModule(SupervisedModule):
    def __init__(self, cfg: StratsModelConfig):
        super(OCPModule, self).__init__(cfg)
        self.cfg = cfg
        if cfg.pretrain:
            assert self.cfg.modalities == SupervisedView.EARLY_FUSION

        # pretrain metrics
        self.train_pretrain_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.train_pretrain_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        self.val_pretrain_acc = torchmetrics.Accuracy(num_classes=cfg.batch_size, task="binary")
        self.val_pretrain_auc = torchmetrics.AUROC(num_classes=cfg.batch_size, task="binary")

        # pretraining model componenets
        self.pretrain_projection = nn.Linear(cfg.embed_size, 1)
        self.pretrain_criterion = torch.nn.BCEWithLogitsLoss()

    def pretrain_forward(self, batch):
        assert self.cfg.modalities == SupervisedView.EARLY_FUSION
        outputs, attn = self.pre_model(batch[SupervisedView.EARLY_FUSION.value])

        logits = self.pretrain_projection(outputs)
        loss = self.pretrain_criterion(logits.squeeze(), batch["flip"].float())

        return OCPPretrainOutput(
            logits=logits,
            loss=loss,
            attn=attn,
            outputs=outputs,
        )

    def pretrain_training_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        # pre metrics
        labels = batch["flip"].float()
        self.train_pretrain_acc.update(output.logits.squeeze(), labels)
        self.train_pretrain_auc.update(output.logits.squeeze(), labels)
        return output

    def pretrain_validation_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        labels = batch["flip"].float()
        self.val_pretrain_acc.update(output.logits.squeeze(), labels)
        self.val_pretrain_auc.update(output.logits.squeeze(), labels)
        return output

    def pretrain_test_step(self, batch):
        output: OCPPretrainOutput = self.forward(batch)
        # pretrain metrics
        labels = batch["flip"].float()
        self.test_pretrain_acc.update(output.logits.squeeze(), labels)
        self.test_pretrain_auc.update(output.logits.squeeze(), labels)
        return output

    def pretrain_on_train_epoch_end(self):
        self.log("train_pretrain_acc", self.train_pretrain_acc, on_epoch=True, batch_size=self.cfg.batch_size)
        self.log("train_pretrain_auc", self.train_pretrain_auc, on_epoch=True, batch_size=self.cfg.batch_size)

    def pretrain_on_val_epoch_end(self):
        self.log("val_pretrain_acc", self.val_pretrain_acc, on_epoch=True, batch_size=self.cfg.batch_size)
        self.log("val_pretrain_auc", self.val_pretrain_auc, on_epoch=True, batch_size=self.cfg.batch_size)
        print("val_pretrain_acc", self.val_pretrain_acc.compute(), "val_pretrain_auc", self.val_pretrain_auc.compute())

    def pretrain_on_test_epoch_end(self):
        self.log("test_pretrain_acc", self.test_pretrain_acc, on_epoch=True, batch_size=self.cfg.batch_size)
        self.log("test_pretrain_auc", self.test_pretrain_auc, on_epoch=True, batch_size=self.cfg.batch_size)
        print("test_pretrain_acc", self.test_pretrain_acc.compute(), "test_pretrain_auc", self.test_pretrain_auc.compute())

    def configure_optimizers(self):
        if self.cfg.half_dtype:
            eps = 1e-4
        else:
            eps = 1e-8
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, eps=eps)
        return optimizer

    @classmethod
    def initialize_pretrain(cls, cfg: StratsModelConfig):
        assert cfg.pretrain
        return cls(cfg)
