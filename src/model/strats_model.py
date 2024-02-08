import dataclasses
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
import torchmetrics
import pytorch_lightning as pl

from src.model.utils import OutputBase
from src.configs.model_configs import StratsModelConfig
from src.configs.dataloader_configs import SupervisedView
from src.model.supervised_model import SupervisedModule

from .architectures.triplet_transformer import TransformerModel


@dataclasses.dataclass
class StratsPretrainOutput(OutputBase):
    loss: torch.Tensor
    forecast: torch.Tensor
    attn: torch.Tensor
    outputs: torch.Tensor


class StratsModule(SupervisedModule):
    def __init__(self, cfg: StratsModelConfig):
        super(StratsModule, self).__init__(cfg)
        self.cfg = cfg
        if cfg.pretrain:
            assert self.cfg.modalities == SupervisedView.EARLY_FUSION

        # pretraining model componenets
        self.forecast_dim = cfg.n_variable - cfg.n_cat_variable + cfg.n_cat_value
        self.forecast_projection = nn.Linear(cfg.embed_size, self.forecast_dim)
        self.forecast_criterion = nn.MSELoss(reduction='none')  # this computes the loss for each element in the batch instead of averaging the squared error.
        # we do this because we will mask out values that are not present in the prediction window and ignore them for the MSE calculation

    def pretrain_forward(self, batch):
        assert self.cfg.modalities == SupervisedView.EARLY_FUSION
        outputs, attn = self.pre_model(batch[SupervisedView.EARLY_FUSION.value])
        forecast_target = batch['forecast_target']
        forecast_target_mask = batch['forecast_target_mask']  # mask 1s are the ones we want to predict
        forecast = self.forecast_projection(outputs)

        loss = self.forecast_criterion(forecast, forecast_target)  # (loss * forecast_target_mask)
        loss = ((loss * forecast_target_mask).T / forecast_target_mask.sum(dim=-1)).sum(dim=0).mean()  # gives mean squred error over unmasked elements

        return StratsPretrainOutput(
            forecast=forecast,
            loss=loss,
            attn=attn,
            outputs=outputs,
        )

    def pretrain_training_step(self, batch):
        output: StratsPretrainOutput = self.forward(batch)
        return output

    def pretrain_validation_step(self, batch):
        output: StratsPretrainOutput = self.forward(batch)
        return output

    def pretrain_test_step(self, batch):
        output: StratsPretrainOutput = self.forward(batch)
        return output

    def pretrain_on_train_epoch_end(self):
        pass

    def pretrain_on_val_epoch_end(self):
        pass

    def pretrain_on_test_epoch_end(self):
        pass

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
