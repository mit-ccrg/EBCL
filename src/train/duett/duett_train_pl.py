import argparse
import copy
import logging
import os
import shutil
from dataclasses import field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Type, TypeVar

import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import torch.optim as optim
from omegaconf import MISSING, OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)

from src.configs.dataloader_configs import CacheMode, DuettDataloaderConfig, SupervisedView
from src.configs.model_configs import DuettModelConfig
from src.configs.train_configs import (
    PRETRAIN_OVERRIDES,
    SUPERVISED_OVERRIDES,
    EncounterSet,
    Experiment,
    LoggerConfig,
    PretrainConfig,
    PretrainMethod,
    SupervisedTrainConfig,
    get_pretrain_exp_config,
    get_supervised_exp_config,
    save_config,
)
from src.configs.utils import hydra_dataclass
from src.data.dataloaders.cached_dataset import CachedDataLoaderCreator
from src.model.duett_model_static import DuettModule, initialize_pretrain_duett

torch.multiprocessing.set_sharing_strategy("file_system")


class WarmUpCallback(pl.callbacks.Callback):
    """Linear warmup over warmup_steps batches, tries to auto-detect the base lr"""

    def __init__(self, steps=1000, base_lr=None, invsqrt=True, decay=None):
        print(
            "warmup_steps {}, base_lr {}, invsqrt {}, decay {}".format(
                steps, base_lr, invsqrt, decay
            )
        )
        self.warmup_steps = steps
        if decay is None:
            self.decay = steps
        else:
            self.decay = decay

        if base_lr is None:
            self.state = {"steps": 0, "base_lr": base_lr}
        else:
            self.state = {"steps": 0, "base_lr": float(base_lr)}

        self.invsqrt = invsqrt

    def set_lr(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        optimizers = model.optimizers()

        if self.state["steps"] < self.warmup_steps:
            if type(optimizers) == "list":
                if self.state["base_lr"] is None:
                    self.state["base_lr"] = [
                        o.param_groups[0]["lr"] for o in optimizers
                    ]
                for opt, base in zip(optimizers, self.state["base_lr"]):
                    self.set_lr(opt, self.state["steps"] / self.warmup_steps * base)
            else:
                if self.state["base_lr"] is None:
                    self.state["base_lr"] = optimizers.param_groups[0]["lr"]
                self.set_lr(
                    optimizers,
                    self.state["steps"] / self.warmup_steps * self.state["base_lr"],
                )
            self.state["steps"] += 1
        elif self.invsqrt:
            if type(optimizers) == "list":
                if self.state["base_lr"] is None:
                    self.state["base_lr"] = [
                        o.param_groups[0]["lr"] for o in optimizers
                    ]
                for opt, base in zip(optimizers, self.state["base_lr"]):
                    self.set_lr(
                        opt,
                        base
                        * (
                            self.decay
                            / (self.state["steps"] - self.warmup_steps + self.decay)
                        )
                        ** 0.5,
                    )
            else:
                if self.state["base_lr"] is None:
                    self.state["base_lr"] = optimizers.param_groups[0]["lr"]
                self.set_lr(
                    optimizers,
                    self.state["base_lr"]
                    * (
                        self.decay
                        / (self.state["steps"] - self.warmup_steps + self.decay)
                    )
                    ** 0.5,
                )
            self.state["steps"] += 1

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


class DuettLightningDataset(pl.LightningDataModule):
    def __init__(self, duett_dataloader_creator: CachedDataLoaderCreator):
        super().__init__()
        self.dataloader_creator = duett_dataloader_creator
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = False

    def setup(self, stage=None):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = self.dataloader_creator.get_dataloaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader


def get_trainer(cfg, use_ray: bool, warmup_steps=2000):
    warmup = WarmUpCallback(steps=warmup_steps)
    if use_ray:
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=[
                warmup,
                RayTrainReportCallback(),
                EarlyStopping(monitor="val_loss", mode="min"),
            ],
            plugins=[RayLightningEnvironment()],
            enable_progress_bar=True,
            num_sanity_val_steps=2,
            max_epochs=cfg.max_epochs,
            gradient_clip_val=1.0,
        )
        trainer = prepare_trainer(trainer)
    else:
        ckpt_dir = os.path.join(cfg.get_output_path(), "checkpoints")
        checkpoint = pl.callbacks.ModelCheckpoint(
            save_last=True,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=ckpt_dir,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            num_nodes=1,
            num_sanity_val_steps=2,
            max_epochs=cfg.max_epochs,
            gradient_clip_val=1.0,
            callbacks=[
                warmup,
                checkpoint,
                EarlyStopping(monitor="val_loss", mode="min"),
            ],
        )
    return trainer


def train_duett(
    cfg: PretrainConfig,
    use_ray: bool,  # set true when using ray tune, otherwise make false to use logger_config.save_dir
    hparams: Mapping[str, Any],
):
    logging.info(f"Saving to {cfg.get_output_path()}")
    assert len(hparams) == 3, f"Extra hparam keys passed: {hparams.keys()}"
    cfg = copy.deepcopy(cfg)
    cfg.model_config.lr = hparams["lr"]
    cfg.model_config.dropout = hparams["dropout"]
    seed = hparams["seed"]
    pl.seed_everything(seed)

    logging.info("Loading Dataset...")
    duett_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = DuettLightningDataset(duett_dataloader_creator)
    dm.setup()

    logging.info("Initialize Model...")
    pretrain_model = initialize_pretrain_duett(cfg.model_config)
    if cfg.model_config.half_dtype:
        pretrain_model.half()

    torch.set_float32_matmul_precision("medium")
    trainer = get_trainer(cfg, use_ray)

    logging.info("Traning Model...")
    trainer.fit(pretrain_model, dm)


defaults = [
    {"dataloader_config": "duett"},
    {"model_config": "duett"},
    "_self_",  # put this last so that we override PretrainConfig with these defaults^
]


@hydra_dataclass
class DuettPretrainTuneConfig(PretrainConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    pretrain_method: PretrainMethod = PretrainMethod.DUETT
    batch_size: int = 256
    logger_config: LoggerConfig = LoggerConfig(
        save_dir="/storage/x-x-x/results/duett_pretrain", name="ray_duett_pretrain"
    )
    dataloader_config: DuettDataloaderConfig = DuettDataloaderConfig(
        cache_mode=CacheMode.LOAD_CACHE,
        num_cpus=8,
        encounter_set=EncounterSet.SUFFICIENT,
        half_dtype=False,
        num_events=128,
    )
    model_config: DuettModelConfig = DuettModelConfig(
        pretrain=True,
        half_dtype=False,
        num_event=128,
    )  # pretraining should be true
    name: str = "ray_duett_pretrain"
    num_samples: int = 16
    max_epochs: int = 300
    grace_period: int = 4
    gpu_ids: str = "0,"

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.dataloader_config, DuettDataloaderConfig):
            assert self.dataloader_config.modalities != SupervisedView.EARLY_FUSION
            if isinstance(self.model_config, DuettModelConfig):
                assert self.model_config.num_event == self.dataloader_config.num_events

    def get_output_path(self):
        suffix = ""
        if self.dataloader_config.modalities != SupervisedView.PRE_AND_POST:
            suffix += f"_{self.dataloader_config.modalities.value}"
        return self.logger_config.save_dir + f"/pretrain_numevent{self.model_config.num_event}" + suffix


@hydra.main(version_base=None, config_name="duett_pretrain_tune_config")
def duett_train_app(cfg: DuettPretrainTuneConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logging.info("Setting up config...")
    assert cfg.pretrain_method == PretrainMethod.DUETT
    assert cfg.model_config.pretrain is True
    assert (
        cfg.dataloader_config.encounter_set == EncounterSet.SUFFICIENT
    )  # should run on whole dataset
    assert cfg.model_config.half_dtype == cfg.dataloader_config.half_dtype
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")
    train_duett(
        cfg,
        False,
        {
            "lr": cfg.model_config.lr,
            "dropout": cfg.model_config.dropout,
            "seed": cfg.seed,
        },
    )


if __name__ == "__main__":
    # python -m src.train.duett.duett_train_pl name=test_duett_pretrain logger_config.save_dir=/storage/x-x-x/results/test_duett_pretrain gpu_ids="0," dataloader_config.modalities=PRE_AND_POST
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    duett_train_app()
