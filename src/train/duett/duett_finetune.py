import copy
import logging
from dataclasses import field
from typing import Any, List, Mapping
import os

import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing

from src.configs.dataloader_configs import CacheMode, DuettDataloaderConfig, SupervisedView
from src.configs.model_configs import DuettModelConfig
from src.configs.train_configs import (
    EncounterSet,
    LoggerConfig,
    PretrainMethod,
    SupervisedMethod,
    SupervisedTrainConfig,
    save_config,
)
from src.configs.utils import hydra_dataclass
from src.data.dataloaders.cached_dataset import CachedDataLoaderCreator
from src.model.duett_model_static import load_fine_tune_duett
from src.train.duett.duett_train_pl import DuettLightningDataset, get_trainer

torch.multiprocessing.set_sharing_strategy("file_system")


defaults = [
    {"dataloader_config": "duett"},
    {"model_config": "duett"},
    "_self_",  # put this last so that we override PretrainConfig with these defaults^
]


@hydra_dataclass
class DuettFinetuneConfig(SupervisedTrainConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    pretrain_method: PretrainMethod = PretrainMethod.DUETT
    supervised_method: SupervisedMethod = SupervisedMethod.DUETT
    batch_size: int = 256
    logger_config: LoggerConfig = LoggerConfig(
        save_dir="/storage/x-x-x/results/duett_finetune", name="ray_duett_finetune"
    )
    dataloader_config: DuettDataloaderConfig = DuettDataloaderConfig(
        cache_mode=CacheMode.LOAD_CACHE,
        num_cpus=8,
    )
    pretrain_ckpt: str = "/storage/x-x-x/results/duett_pretrain/best_checkpoint/checkpoint.ckpt"
    model_config: DuettModelConfig = DuettModelConfig(pretrain=False)
    name: str = "ray_duett_finetune"
    num_samples: int = 16
    max_epochs: int = 50
    grace_period: int = 4
    gpu_ids: str = "0,"
    
    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.dataloader_config, DuettDataloaderConfig):
            assert self.dataloader_config.modalities != SupervisedView.EARLY_FUSION

    def get_output_path(self):
        suffix = ""
        if self.dataloader_config.modalities != SupervisedView.PRE_AND_POST:
            suffix += f"_{self.dataloader_config.modalities.value}"
        return self.logger_config.save_dir + f"/{self.dataloader_config.encounter_set.value}" + f"_numevent{self.model_config.num_event}" + suffix


def finetune_duett(
    cfg: DuettFinetuneConfig,
    use_ray: bool,  # set true when using ray tune, otherwise make false to use logger_config.save_dir
    hparams: Mapping[str, Any],
):
    assert len(hparams) == 3, f"Incorrect number of hparam keys passed: {hparams.keys()}"
    cfg = copy.deepcopy(cfg)
    cfg.model_config.lr = hparams["lr"]
    cfg.model_config.dropout = hparams["dropout"]
    seed = hparams["seed"]

    pl.seed_everything(seed)

    logging.info("Loading Dataset...")
    duett_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = DuettLightningDataset(duett_dataloader_creator)
    dm.setup()

    logging.info("Loading Pretrained Model...")
    checkpoint_pth = cfg.pretrain_ckpt
    finetune_model = load_fine_tune_duett(cfg.model_config, checkpoint_pth)

    assert not finetune_model.pretrain

    torch.set_float32_matmul_precision("medium")
    trainer = get_trainer(cfg, use_ray, warmup_steps=1000)

    logging.info("Traning Model...")
    trainer.fit(finetune_model, dm)

    # results = trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path, datamodule=dm)
    # assert len(results) == 1
    # return results[0] # {'test_loss': 0.5327323079109192, 'test_auroc': 0.7647244930267334, 'test_acc': 0.7273656129837036, 'test_ap': 0.8746918439865112}


@hydra.main(version_base=None, config_name="duett_finetune_config")
def duett_finetune_app(cfg: DuettFinetuneConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logging.info("Setting up config...")
    assert cfg.pretrain_method == PretrainMethod.DUETT
    assert cfg.supervised_method == SupervisedMethod.DUETT
    assert cfg.model_config.pretrain is False
    assert cfg.dataloader_config.encounter_set in [
        EncounterSet.READMISSION,
        EncounterSet.MORTALITY,
    ]  # should run on whole dataset

    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")
    finetune_duett(
        cfg,
        False,
        {
            "lr": cfg.model_config.lr,
            "dropout": cfg.model_config.dropout,
            "seed": cfg.seed,
        },
    )


if __name__ == "__main__":
    # python -m src.train.duett_finetune dataloader_config.encounter_set=MORTALITY model_config.lr=1e-4 model_config.dropout=0.5 seed=0 logger_config.name=test_finetune_duett max_epochs=2
    # ls /storage/x-x-x/results/duett_pretrain/ray_duett_pretrain/TorchTrainer_8ee0e_00001_1_dropout=0.5341,lr=0.0009,seed=2016_2023-12-19_23-30-52/checkpoint_000033/checkpoint.ckpt
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    duett_finetune_app()
