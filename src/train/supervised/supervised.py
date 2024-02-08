import copy
import logging
from dataclasses import field
from typing import Any, List, Mapping, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import os

from src.configs.dataloader_configs import CacheMode, EBCLDataloaderConfig, SupervisedView, EBCLSamplingMethod
from src.configs.model_configs import StratsModelConfig
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
from src.model.ebcl_model import load_fine_tune_ebcl, EBCLModule
from src.train.ebcl.ebcl_train_pl import EBCLLightningDataset, get_trainer

torch.multiprocessing.set_sharing_strategy("file_system")


defaults = [
    {"dataloader_config": "ebcl"},
    {"model_config": "ebcl"},
    "_self_",  # put this last so that we override PretrainConfig with these defaults^
]


@hydra_dataclass
class EbclFinetuneConfig(SupervisedTrainConfig):
    defaults: List[Any] = field(default_factory=lambda: defaults)
    pretrain_method: PretrainMethod = PretrainMethod.EBCL
    supervised_method: SupervisedMethod = SupervisedMethod.EBCL
    batch_size: int = 256
    logger_config: LoggerConfig = LoggerConfig(
        save_dir="/storage/x-x-x/results/ebcl_finetune", name="ray_ebcl_finetune"
    )
    dataloader_config: EBCLDataloaderConfig = EBCLDataloaderConfig(
        cache_mode=CacheMode.LOAD_CACHE,
        num_cpus=8,
    )
    pretrain_ckpt: Optional[str] = None
    model_config: StratsModelConfig = StratsModelConfig(pretrain=False)
    name: str = "ray_ebcl_finetune"
    num_samples: int = 1
    max_epochs: int = 100
    grace_period: int = 4
    modalities: SupervisedView = SupervisedView.PRE_AND_POST
    half_dtype: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.pretrain_ckpt is None:
            logging.info("Supervised Finetuning without Pretraining")
        if isinstance(self.dataloader_config, EBCLDataloaderConfig):
            self.dataloader_config.modalities = self.modalities
            self.dataloader_config.half_dtype = self.half_dtype
            assert self.dataloader_config.encounter_set != EncounterSet.SUFFICIENT  # should be set to a task specific encounter set
            assert self.dataloader_config.sampling_method == EBCLSamplingMethod.ADMISSION
        if isinstance(self.model_config, StratsModelConfig):
            assert self.model_config.pretrain is False
            self.model_config.modalities = self.modalities
            self.model_config.half_dtype = self.half_dtype

    def get_output_path(self):
        suffix = "/embed_size_finetune"
        suffix += f"_{self.dataloader_config.modalities.value}"
        if self.half_dtype:
            suffix += "_halfdtype"
        if self.pretrain_ckpt is not None:
            suffix += f"_{self.dataloader_config.sampling_method.value}ebcl"
        else:
            assert self.dataloader_config.sampling_method == EBCLSamplingMethod.ADMISSION
            suffix += "_supervised"
        suffix += f"_{self.dataloader_config.encounter_set.value}"
        output_path = self.logger_config.save_dir + suffix
        logging.warn(f"Saving to {output_path}")
        return output_path


def finetune_ebcl(
    cfg: EbclFinetuneConfig,
    use_ray: bool,  # set true when using ray tune, otherwise make false to use logger_config.save_dir
    hparams: Mapping[str, Any],
):
    assert len(hparams) == 2, f"Incorrect number of hparam keys passed: {hparams.keys()}"
    cfg = copy.deepcopy(cfg)
    assert isinstance(cfg.model_config, StratsModelConfig)
    cfg.model_config.embed_size = hparams["embed_size"]
    cfg.model_config.projection_dim = hparams["embed_size"]
    cfg.model_config.encoder_dim = hparams["embed_size"]
    seed = hparams["seed"]

    pl.seed_everything(seed)

    logging.info("Loading Dataset...")
    ebcl_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = EBCLLightningDataset(ebcl_dataloader_creator)
    dm.setup()

    logging.info("Loading Pretrained Model...")
    checkpoint_pth = cfg.pretrain_ckpt
    if cfg.pretrain_ckpt is None:
        finetune_model = EBCLModule(cfg=cfg.model_config)
    else:
        finetune_model = load_fine_tune_ebcl(cfg.model_config, checkpoint_pth)
    if cfg.model_config.half_dtype:
        logging.warning(f"Using Half Precision ->")
        finetune_model.half()  # half precision

    assert not finetune_model.cfg.pretrain

    torch.set_float32_matmul_precision("medium")
    trainer = get_trainer(cfg, use_ray)

    logging.info("Traning Model...")
    trainer.fit(finetune_model, dm)

    # results = trainer.test(ckpt_path=trainer.checkpoint_callback.best_model_path, datamodule=dm)
    # assert len(results) == 1
    # return results[0] # {'test_loss': 0.5327323079109192, 'test_auroc': 0.7647244930267334, 'test_acc': 0.7273656129837036, 'test_ap': 0.8746918439865112}


@hydra.main(version_base=None, config_name="ebcl_finetune_config")
def ebcl_finetune_app(cfg: EbclFinetuneConfig):
    logging.info("Setting up config...")
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    assert cfg.model_config.pretrain is False
    assert cfg.dataloader_config.encounter_set in [
        EncounterSet.READMISSION,
        EncounterSet.MORTALITY,
        EncounterSet.LOS
    ]  # should run on whole dataset
    if cfg.dataloader_config.encounter_set == EncounterSet.LOS:
        assert cfg.dataloader_config.modalities == SupervisedView.PRE

    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")
    finetune_ebcl(
        cfg,
        False,
        {
            "nheads": cfg.model_config.nheads,
            "nlayers": cfg.model_config.nlayers,
            "embed_size": cfg.model_config.embed_size,
            "seed": cfg.seed,
        },
    )


if __name__ == "__main__":
    # Test supervised finetuning
    # python -m src.train.supervised.supervised name=test_supervised max_epochs=2 batch_size=256 dataloader_config.num_cpus=4 half_dtype=True modalities=PRE_AND_POST dataloader_config.encounter_set=MORTALITY
    os.environ['HYDRA_FULL_ERROR'] = "1"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    ebcl_finetune_app()
