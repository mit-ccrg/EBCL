import dataclasses
import enum
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from src.model.utils import EncoderBase, OutputBase
from src.configs.model_configs import StratsModelConfig
from src.configs.dataloader_configs import SupervisedView

from .architectures.triplet_transformer import TransformerModel


@dataclasses.dataclass
class SupervisedOutput(OutputBase):
    final_output: torch.Tensor
    post_embeds: torch.Tensor
    pre_embeds: torch.Tensor
    post_model_output: torch.Tensor
    pre_model_output: torch.Tensor
    pre_attn: torch.Tensor
    post_attn: torch.Tensor
    loss: torch.Tensor


class SupervisedModule(pl.LightningModule):
    def __init__(self, cfg: StratsModelConfig):
        super(SupervisedModule, self).__init__()
        self.cfg = cfg
        # shared components
        self.pre_model = TransformerModel(self.cfg)
        if cfg.shared_transformer:
            self.post_model = self.pre_model
        else:
            self.post_model = TransformerModel(self.cfg)
            raise ValueError("Shared Transformer should not be used for current Experiments")

        self.post_projection = nn.Linear(cfg.embed_size, cfg.projection_dim)
        self.pre_projection = nn.Linear(cfg.embed_size, cfg.projection_dim)

        # finetune metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.train_apr = torchmetrics.AveragePrecision(task="binary")

        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.val_apr = torchmetrics.AveragePrecision(task="binary")

        self.test_acc = torchmetrics.Accuracy(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        self.test_apr = torchmetrics.AveragePrecision(task="binary")

        if cfg.bootstrap:
            self.test_bootstrap_acc = torchmetrics.wrappers.BootStrapper(torchmetrics.Accuracy(task="binary"), num_bootstraps=100)
            self.test_bootstrap_auc = torchmetrics.wrappers.BootStrapper(torchmetrics.AUROC(task="binary"), num_bootstraps=100)
            self.test_bootstrap_apr = torchmetrics.wrappers.BootStrapper(torchmetrics.AveragePrecision(task="binary"), num_bootstraps=100)

        # finetuning model components
        self.class_projection = nn.Linear(2 * cfg.projection_dim, 1)  # use for multimodal
        self.unimodal_class_projection = nn.Linear(cfg.projection_dim, 1)  # use for unimodal
        self.finetune_criterion = torch.nn.BCEWithLogitsLoss()

    def pretrain_forward(self, batch):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def finetune_forward(self, batch, get_representations=False):
        pre_embeds, post_embeds = None, None

        if self.cfg.modalities == SupervisedView.POST:
            pre_outputs, pre_attn = None, None
            post_outputs, post_attn = self.post_model(batch[SupervisedView.POST.value])
            post_embeds = F.relu(self.post_projection(post_outputs))
            post_embeds = F.dropout(
                post_embeds, p=self.cfg.dropout, training=self.training
            )
            input_fc = post_embeds
            final_output = self.unimodal_class_projection(input_fc)
        elif self.cfg.modalities == SupervisedView.PRE:
            post_outputs, post_attn = None, None
            pre_outputs, pre_attn = self.pre_model(batch[SupervisedView.PRE.value])
            pre_embeds = F.relu(self.pre_projection(pre_outputs))
            pre_embeds = F.dropout(
                pre_embeds, p=self.cfg.dropout, training=self.training
            )
            input_fc = pre_embeds
            final_output = self.unimodal_class_projection(input_fc)
        elif self.cfg.modalities == SupervisedView.EARLY_FUSION:
            post_outputs, post_attn = None, None
            pre_outputs, pre_attn = self.pre_model(batch[SupervisedView.EARLY_FUSION.value])
            pre_embeds = F.relu(self.pre_projection(pre_outputs))
            pre_embeds = F.dropout(
                pre_embeds, p=self.cfg.dropout, training=self.training
            )
            input_fc = pre_embeds
            final_output = self.unimodal_class_projection(input_fc)
        else:
            assert (
                self.cfg.modalities == SupervisedView.PRE_AND_POST
            ), f"self.cfg.modalities {self.cfg.modalities}"
            pre_outputs, pre_attn = self.pre_model(batch[SupervisedView.PRE.value])
            post_outputs, post_attn = self.post_model(batch[SupervisedView.POST.value])
            pre_embeds = F.relu(self.pre_projection(pre_outputs))
            post_embeds = F.relu(self.post_projection(post_outputs))
            concat_embeds = torch.cat((pre_embeds, post_embeds), dim=1)
            concat_embeds = F.dropout(
                concat_embeds, p=self.cfg.dropout, training=self.training
            )
            input_fc = concat_embeds
            final_output = self.class_projection(input_fc)
        if get_representations:
            loss = None
        else:
            loss = self.finetune_criterion(final_output.squeeze(), batch["outcome"].float())
        return SupervisedOutput(
                final_output=final_output,
                post_embeds=post_embeds,
                pre_embeds=pre_embeds,
                post_model_output=post_outputs,
                pre_model_output=pre_outputs,
                pre_attn=pre_attn,
                post_attn=post_attn,
                loss=loss,
            )

    def forward(self, batch) -> OutputBase:
        if self.cfg.pretrain:
            return self.pretrain_forward(batch)
        else:
            return self.finetune_forward(batch)

    def pretrain_training_step(self, batch):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def pretrain_validation_step(self, batch):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def pretrain_test_step(self, batch):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def pretrain_on_train_epoch_end(self):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def pretrain_on_val_epoch_end(self):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def pretrain_on_test_epoch_end(self):
        raise NotImplementedError(f"Pretraining not implemented for {self.__name__}")

    def training_step(self, batch):
        if self.cfg.pretrain:
            output = self.pretrain_training_step(batch)
        else:
            output: SupervisedOutput = self.forward(batch)
            # logs metrics for each training_step,
            # and the average across the epoch
            # finetune metrics
            self.train_acc.update(output.final_output.squeeze(), batch["outcome"].float())
            self.train_auc.update(output.final_output.squeeze(), batch["outcome"].float())
            self.train_apr.update(output.final_output.squeeze(), batch["outcome"].int())
        self.log("train_loss", output.loss, batch_size=self.cfg.batch_size)
        assert not torch.isnan(output.loss), "Loss is NaN"
        return output.loss

    def on_train_epoch_end(self):
        if self.cfg.pretrain:
            self.pretrain_on_train_epoch_end()
        else:
            self.log("train_auc", self.train_auc, on_epoch=True, batch_size=self.cfg.batch_size)
            self.log("train_acc", self.train_acc, on_epoch=True, batch_size=self.cfg.batch_size)
            self.log("train_apr", self.train_apr, on_epoch=True, batch_size=self.cfg.batch_size)

    def validation_step(self, batch):
        if self.cfg.pretrain:
            output = self.pretrain_validation_step(batch)
        else:
            output: OutputBase = self.forward(batch)
            # logs metrics for each training_step,
            # and the average across the epoch
            # finetune metrics
            self.val_acc.update(output.final_output.squeeze(), batch["outcome"].float())
            self.val_auc.update(output.final_output.squeeze(), batch["outcome"].float())
            self.val_apr.update(output.final_output.squeeze(), batch["outcome"].int())

        self.log("val_loss", output.loss, on_epoch=True, batch_size=self.cfg.batch_size)
        return output.loss

    def on_val_epoch_end(self):
        if self.cfg.pretrain:
            self.pretrain_on_val_epoch_end()
        else:
            self.log("val_auc", self.val_auc, on_epoch=True, batch_size=self.cfg.batch_size)
            self.log("val_acc", self.val_acc, on_epoch=True, batch_size=self.cfg.batch_size)
            self.log("val_apr", self.val_apr, on_epoch=True, batch_size=self.cfg.batch_size)
            print("val_auc", self.val_auc.compute(), "val_acc", self.val_acc.compute(), "val_apr", self.val_apr.compute())

    def test_step(self, batch, batch_idx):
        if self.cfg.pretrain:
            output = self.pretrain_test_step(batch)
        else:
            output: OutputBase = self.forward(batch)
            # logs metrics for each training_step,
            # and the average across the epoch
            # finetune metrics
            self.test_acc.update(output.final_output.squeeze(), batch["outcome"].float())
            self.test_auc.update(output.final_output.squeeze(), batch["outcome"].float())
            self.test_apr.update(output.final_output.squeeze(), batch["outcome"].int())

            if self.cfg.bootstrap:
                self.test_bootstrap_acc .update(output.final_output.squeeze(), batch["outcome"].float())
                self.test_bootstrap_auc.update(output.final_output.squeeze(), batch["outcome"].float())
                self.test_bootstrap_apr.update(output.final_output.squeeze(), batch["outcome"].int())
        self.log("test_loss", output.loss, batch_size=self.cfg.batch_size)
        return output.loss

    def on_test_epoch_end(self):
        if self.cfg.pretrain:
            self.pretrain_on_test_epoch_end()
        else:
            self.log("test_auc", self.test_auc, on_epoch=True, batch_size=self.cfg.batch_size)
            self.log("test_acc", self.test_acc, on_epoch=True, batch_size=self.cfg.batch_size)
            self.log("test_apr", self.test_apr, on_epoch=True, batch_size=self.cfg.batch_size)
            print("test_auc", self.test_auc.compute(), "test_acc", self.test_acc.compute(), "test_apr", self.test_apr.compute())

            if self.cfg.bootstrap:
                bootstrap_auc = self.test_bootstrap_auc.compute()
                self.log("test_bootstrap_auc_mean", bootstrap_auc['mean'], on_epoch=True, batch_size=self.cfg.batch_size)
                self.log("test_bootstrap_auc_std", bootstrap_auc['std'], on_epoch=True, batch_size=self.cfg.batch_size)

                bootstrap_acc = self.test_bootstrap_auc.compute()
                self.log("test_bootstrap_acc_mean", bootstrap_acc['mean'], on_epoch=True, batch_size=self.cfg.batch_size)
                self.log("test_bootstrap_acc_std", bootstrap_acc['std'], on_epoch=True, batch_size=self.cfg.batch_size)

                bootstrap_apr = self.test_bootstrap_apr.compute()
                self.log("test_bootstrap_apr_mean", bootstrap_apr['mean'], on_epoch=True, batch_size=self.cfg.batch_size)
                self.log("test_bootstrap_apr_std", bootstrap_apr['std'], on_epoch=True, batch_size=self.cfg.batch_size)

    def configure_optimizers(self):
        if self.cfg.half_dtype:
            eps = 1e-4
        else:
            eps = 1e-8
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr, eps=eps)
        return optimizer

    @classmethod
    def initialize_pretrain(cls, cfg):
        raise NotImplementedError(f"Pretraining not implemented for {cls.__name__}")

    @classmethod
    def initialize_finetune(cls, cfg, ckpt_path: str):
        assert isinstance(cfg, StratsModelConfig)
        assert not cfg.pretrain
        return cls.load_from_checkpoint(ckpt_path, cfg=cfg)
