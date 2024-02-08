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

from src.configs.model_configs import StratsModelConfig
from src.configs.dataloader_configs import CacheMode, EBCLDataloaderConfig, EBCLSamplingMethod, SupervisedView
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
from src.data.dataloaders.ebcl_dataloader import EBCLDataLoaderCreator, ABLATION_ENCOUNTERS
from src.model.ebcl_model import EBCLModule
from src.train.utils import train_loop
from src.utils.logger import Logger, TuneLogger
# from tests.test_models import load_dataloader, load_model

torch.multiprocessing.set_sharing_strategy("file_system")


class EBCLLightningDataset(pl.LightningDataModule):
    def __init__(self, ebcl_dataloader_creator: CachedDataLoaderCreator):
        super().__init__()
        self.dataloader_creator = ebcl_dataloader_creator
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


def get_trainer(cfg, use_ray: bool):
    logging.warn(f"Saving to :{cfg.get_output_path()}")
    if use_ray:
        trainer = pl.Trainer(
            devices="auto",
            accelerator="auto",
            strategy=RayDDPStrategy(find_unused_parameters=True),
            callbacks=[
                RayTrainReportCallback(),
                EarlyStopping(monitor="val_loss", mode="min", patience=cfg.early_stop_tol),
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
                checkpoint,
                EarlyStopping(monitor="val_loss", mode="min"),
            ],
        )
    return trainer


def train_ebcl(
    cfg: PretrainConfig,
    use_ray: bool,  # set true when using ray tune, otherwise make false to use logger_config.save_dir
    hparams: Mapping[str, Any],
):
    assert len(hparams) == 3, f"Extra hparam keys passed: {hparams.keys()}"
    cfg = copy.deepcopy(cfg)
    cfg.model_config.lr = hparams["lr"]
    cfg.model_config.dropout = hparams["dropout"]
    seed = hparams["seed"]
    pl.seed_everything(seed)

    # TODO See if Flash attention is faster
    # with torch.backends.cuda.sdp_kernel(**{"enable_math": False, "enable_flash": True, "enable_mem_efficient": False}):
    logging.warning(f"The flash attention implementation is running")
    logging.info("Loading Dataset...")
    if cfg.dataloader_config.cache_mode == CacheMode.NO_CACHE:
        ebcl_dataloader_creator = EBCLDataLoaderCreator(cfg.dataloader_config)
    else:
        assert cfg.dataloader_config.cache_mode == CacheMode.LOAD_CACHE
        ebcl_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = EBCLLightningDataset(ebcl_dataloader_creator)
    dm.setup()

    logging.info("Initialize Model...")
    pretrain_model = EBCLModule.initialize_pretrain(cfg.model_config)
    if cfg.model_config.half_dtype:
        logging.warning(f"Using Half Precision ->")
        pretrain_model.half()  # half precision

    torch.set_float32_matmul_precision("medium")
    trainer = get_trainer(cfg, use_ray)

    logging.info("Training Model...")
    trainer.fit(pretrain_model, dm)


defaults = [
    {"dataloader_config": "ebcl"},
    {"model_config": "ebcl"},
    "_self_",  # put this last so that we override PretrainConfig with these defaults^
]


@hydra_dataclass
class EbclPretrainTuneConfig(PretrainConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    pretrain_method: PretrainMethod = PretrainMethod.EBCL
    logger_config: LoggerConfig = LoggerConfig(
        save_dir="/storage/x-x-x/results/ebcl_pretrain", name="ray_ebcl_pretrain"
    )
    dataloader_config: EBCLDataloaderConfig = EBCLDataloaderConfig(
        cache_mode=CacheMode.LOAD_CACHE,
        num_cpus=8,
        encounter_set=EncounterSet.SUFFICIENT,
        sampling_method=EBCLSamplingMethod.ADMISSION,
    )
    model_config: StratsModelConfig = StratsModelConfig(
        pretrain=True,
        batch_size=256,
    )  # pretraining should be true
    name: str = "ray_ebcl_pretrain"
    num_samples: int = 16
    max_epochs: int = 100
    grace_period: int = 4
    modalities: SupervisedView = SupervisedView.PRE_AND_POST
    half_dtype: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.modalities == SupervisedView.PRE_AND_POST  # EBCL Pretraining requires both pre and post
        if isinstance(self.dataloader_config, EBCLDataloaderConfig):
            self.dataloader_config.modalities = self.modalities
            self.dataloader_config.half_dtype = self.half_dtype
            assert self.dataloader_config.encounter_set == EncounterSet.SUFFICIENT
            if self.dataloader_config.sampling_method != EBCLSamplingMethod.ADMISSION:
                self.dataloader_config.cache_mode = CacheMode.NO_CACHE  # Don't use cache if not doing ADMISSION
            if self.dataloader_config.sampling_method in ABLATION_ENCOUNTERS:
                assert self.dataloader_config.post_los_cutoff is False
            elif self.dataloader_config.sampling_method == EBCLSamplingMethod.CENSORED:
                assert self.dataloader_config.post_los_cutoff is False
            elif self.dataloader_config.sampling_method == EBCLSamplingMethod.RANDOM:
                 assert self.dataloader_config.post_los_cutoff is False
                ## TODO: assign self.dataloader_config.post_los_cutoff to be default False then we don't need to check this here.
            else:
                assert self.dataloader_config.post_los_cutoff is True
        if isinstance(self.model_config, StratsModelConfig):
            assert self.model_config.pretrain is True
            self.model_config.modalities = self.modalities
            self.model_config.half_dtype = self.half_dtype

    def get_output_path(self):
        if self.model_config.pretrain:
            suffix = "/pretrain"
        else:
            suffix = "/finetune"
        suffix += f"_{self.model_config.architecture_size.value}"
        suffix += f"_{self.dataloader_config.modalities.value}"
        if self.half_dtype:
            suffix += "_halfdtype"
        suffix += f"_{self.dataloader_config.sampling_method.value}ebcl"
        self.logger_config.name = suffix[1:]
        return self.logger_config.save_dir + suffix


@hydra.main(version_base=None, config_name="ebcl_pretrain_tune_config")
def ebcl_train_app(cfg: EbclPretrainTuneConfig):
    logging.info("Setting up config...")
    assert cfg.pretrain_method == PretrainMethod.EBCL
    assert cfg.model_config.pretrain is True
    assert (
        cfg.dataloader_config.encounter_set == EncounterSet.SUFFICIENT
    )  # should run on whole dataset
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    assert cfg.batch_size == cfg.model_config.batch_size
    assert cfg.batch_size == cfg.dataloader_config.batch_size
    assert cfg.dataloader_config.half_dtype == cfg.model_config.half_dtype
    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")
    train_ebcl(
        cfg,
        False,
        {
            "lr": cfg.model_config.lr,
            "dropout": cfg.model_config.dropout,
            "seed": cfg.seed,
        },
    )


if __name__ == "__main__":
    # Test ebcl CENSORED pretraining 
    # python -m src.train.ebcl.ebcl_train_pl name=test_ebcl_pretrain max_epochs=2 batch_size=256 dataloader_config.num_cpus=4 half_dtype=False dataloader_config.sampling_method=CENSORED logger_config.save_dir='storage/x-x-x/results/ebcl_pretrain'
    # Test ebcl OUTPATIENT pretraining
    # python -m src.train.ebcl.ebcl_train_pl name=test_ebcl_pretrain max_epochs=2 dataloader_config.num_cpus=4 half_dtype=True dataloader_config.sampling_method=OUTPATIENT dataloader_config.post_los_cutoff=False
    # Test ebcl admission pretraining
    # python -m src.train.ebcl.ebcl_train_pl name=test_ebcl_pretrain max_epochs=2 batch_size=256 dataloader_config.num_cpus=4 half_dtype=True
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    ebcl_train_app()
