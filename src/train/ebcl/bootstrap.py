import copy
import logging
from dataclasses import field
from typing import Any, List, Mapping, Optional

import hydra
import pytorch_lightning as pl
import torch
import torch.multiprocessing
import os
import pandas as pd

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
from src.model.ebcl_model import EBCLModule
from src.train.ebcl.ebcl_train_pl import EBCLLightningDataset, get_trainer
from src.train.ebcl.ebcl_finetune import EbclFinetuneConfig
import os
from src.configs.dataloader_configs import SupervisedView, EBCLSamplingMethod
from src.configs.train_configs import EncounterSet
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
from src.configs.dataloader_configs import SupervisedView, EBCLSamplingMethod
from src.configs.train_configs import EncounterSet
from omegaconf import OmegaConf
from tqdm.auto import trange


#admission
def get_result(dir, name, task, modalities):
    df = pd.read_csv(dir)
    df['exp_name'] = [name]
    df['task'] = [task]
    df['modalities'] = [modalities]
    df['dir'] = [str(Path(dir).parent)]
    df = df[['exp_name', 'task', 'modalities', 'test_auc', 'test_apr', 'test_acc', 'test_loss', 'dir']]
    df[['test_auc', 'test_apr', 'test_acc']] = df[['test_auc', 'test_apr', 'test_acc']] * 100
    df[['test_auc', 'test_apr', 'test_acc', 'test_loss']] = df[['test_auc', 'test_apr', 'test_acc', 'test_loss']].round(2)
    return df

def extract_text_between(s, start_delimiter, end_delimiter):
    """
    Extracts a substring from a string s that is between two delimiter strings.

    :param s: The string to extract from.
    :param start_delimiter: The starting delimiter string.
    :param end_delimiter: The ending delimiter string.
    :return: The substring found between the two delimiters. If the delimiters are not found, returns an empty string.
    """
    start_index = s.find(start_delimiter)
    if start_index == -1:
        return ""  # Start delimiter not found

    start_index += len(start_delimiter)
    end_index = s.find(end_delimiter, start_index)
    if end_index == -1:
        return ""  # End delimiter not found

    return s[start_index:end_index]

def search_files(start_path, file_start):
    """
    Search for files in a directory that start with a specific string.

    :param start_path: The path of the directory to search in.
    :param file_start: The starting string of the file names to look for.
    :return: A list of file paths that start with the specified string.
    """
    matching_files = []
    for root, dirs, files in os.walk(start_path):
        for file in files:
            if file.startswith(file_start):
                full_path = os.path.join(root, file)
                matching_files.append(full_path)
    return matching_files

def get_ebcl_results(exp_dir):
    dfs = []
    for dir in os.listdir(exp_dir):
        split_dir = dir.split("_")
        if len(split_dir) <= 1:
            continue
        architecture = split_dir[1]
        if architecture == "medium":
            task = split_dir[-1]
            exp_name = extract_text_between(dir, 'admissionebcl', task)[1:-1]
            modalities = extract_text_between(dir, 'medium', 'admissionebcl')[1:-1]
            if exp_name == "":
                exp_name = "supervised"
                assert exp_name in dir
                modalities = extract_text_between(dir, 'medium', exp_name)[1:-1]
            results_files = search_files(exp_dir + dir, "test_results_seed")
            if len(results_files) != 1:
                print(f'failed for {exp_dir + dir}')
            else:
                df = get_result(results_files[0], exp_name, task, modalities)
                dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)

torch.multiprocessing.set_sharing_strategy("file_system")


defaults = [
    {"dataloader_config": "ebcl"},
    {"model_config": "ebcl"},
    "_self_",  # put this last so that we override PretrainConfig with these defaults^
]


@hydra_dataclass
class EbclBootstrapConfig(EbclFinetuneConfig):
    logger_config: LoggerConfig = LoggerConfig(
        save_dir="/storage/x-x-x/results/bootstrap/ebcl/", name="ray_ebcl_test"
    )
    model_config: StratsModelConfig = StratsModelConfig(pretrain=False, bootstrap=True)
    name: str = "ray_ebcl_test"
    hyperparam_results_path: str = "/storage/x-x-x/results/ebcl_finetune/"

    def __post_init__(self):
        # super().__post_init__()
        assert self.pretrain_ckpt is not None
        if isinstance(self.dataloader_config, EBCLDataloaderConfig):
            self.dataloader_config.modalities = self.modalities
            self.dataloader_config.half_dtype = self.half_dtype
            assert self.dataloader_config.encounter_set != EncounterSet.SUFFICIENT  # should be set to a task specific encounter set
            assert self.dataloader_config.sampling_method == EBCLSamplingMethod.ADMISSION
        if isinstance(self.model_config, StratsModelConfig):
            assert self.model_config.pretrain is False
            self.model_config.modalities = self.modalities
            self.model_config.half_dtype = self.half_dtype
            assert self.model_config.bootstrap

    def get_output_path(self):
        if self.half_dtype:
            raise ValueError("Half Precision not supported for bootstrap")
        assert self.dataloader_config.sampling_method == EBCLSamplingMethod.ADMISSION
        dir_name = self.pretrain_ckpt.split("/")[-3]
        output_path = self.logger_config.save_dir + dir_name + "/"
        logging.warn(f"Saving to {output_path}")
        return output_path


def test_ebcl(
    cfg: EbclBootstrapConfig,
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
    ebcl_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = EBCLLightningDataset(ebcl_dataloader_creator)
    dm.setup()

    logging.info("Loading Pretrained Model...")
    checkpoint_pth = cfg.pretrain_ckpt
    if cfg.pretrain_ckpt is not None:
        finetune_model = EBCLModule(cfg=cfg.model_config)
    else:
        finetune_model = EBCLModule.initialize_finetune(cfg.model_config, checkpoint_pth)
    if cfg.model_config.half_dtype:
        logging.warning(f"Using Half Precision ->")
        finetune_model.half()  # half precision

    assert not finetune_model.cfg.pretrain

    torch.set_float32_matmul_precision("medium")
    trainer = get_trainer(cfg, use_ray)

    results = trainer.test(ckpt_path=cfg.pretrain_ckpt, datamodule=dm, model=finetune_model)
    output_path = cfg.get_output_path()
    assert 'bootstrap' in output_path
    pd.DataFrame(results).to_csv(os.path.join(output_path, f"test_results.csv"))
    return results[0]


@hydra.main(version_base=None, config_name="ebcl_bootstrap_config")
def ebcl_finetune_app(cfg: EbclBootstrapConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logging.info("Setting up config...")
    ebcl_results = get_ebcl_results(cfg.hyperparam_results_path)
    ebcl_results['bootstrap_auc_mean'] = None
    ebcl_results['bootstrap_auc_std'] = None
    ebcl_results['bootstrap_apr_mean'] = None
    ebcl_results['bootstrap_apr_std'] = None
    ebcl_results['bootstrap_acc_mean'] = None
    ebcl_results['bootstrap_acc_std'] = None
    for i in trange(ebcl_results.shape[0], position=0, leave=True):
        result = ebcl_results.iloc[i]
        try:
            pretrain_sampling = getattr(EBCLSamplingMethod, result.exp_name.upper())
        except:
            pretrain_sampling = None

        # load modalities, encounter_set, and pretrain checkpoint into config
        modalities = getattr(SupervisedView, result.modalities.upper())
        encounter_set = getattr(EncounterSet, result.task.upper())
        pretrain_ckpt = result.dir + "/best_checkpoint/checkpoint.ckpt"
        # assert (pretrain_ckpt is None) == (pretrain_sampling is None)  # either both are None (for supervised experiments) or both are defined (EBCL pretrained experiments)
        cfg.pretrain_ckpt = pretrain_ckpt.replace("server_results", "results")
        cfg.dataloader_config.encounter_set = encounter_set
        cfg.modalities = modalities
        cfg.model_config.modalities = modalities
        cfg.dataloader_config.modalities = modalities
        cfg.pretrain_sampling = pretrain_sampling
        init_cfg = hydra.utils.instantiate(cfg, _convert_="object")
        assert init_cfg.model_config.pretrain is False
        assert init_cfg.dataloader_config.encounter_set in [
            EncounterSet.READMISSION,
            EncounterSet.MORTALITY,
            EncounterSet.LOS
        ]  # should run on whole dataset
        if init_cfg.dataloader_config.encounter_set == EncounterSet.LOS:
            assert init_cfg.dataloader_config.modalities == SupervisedView.PRE
        output_path = init_cfg.get_output_path()
        os.makedirs(output_path, exist_ok=True)
        save_config(init_cfg, output_path + "/config.yml")
        test_result = test_ebcl(
            init_cfg,
            False,
            {
                "lr": init_cfg.model_config.lr,
                "dropout": init_cfg.model_config.dropout,
                "seed": init_cfg.seed,
            },
        )
        ebcl_results.at[i, 'bootstrap_auc_mean'] = test_result['test_bootstrap_auc_mean']
        ebcl_results.at[i, 'bootstrap_auc_std'] = test_result['test_bootstrap_auc_std']
        ebcl_results.at[i, 'bootstrap_apr_mean'] = test_result['test_bootstrap_apr_mean']
        ebcl_results.at[i, 'bootstrap_apr_std'] = test_result['test_bootstrap_apr_std']
        ebcl_results.at[i, 'bootstrap_acc_mean'] = test_result['test_bootstrap_acc_mean']
        ebcl_results.at[i, 'bootstrap_acc_std'] = test_result['test_bootstrap_acc_std']
        ebcl_results.to_csv(os.path.join(cfg.logger_config.save_dir, f"bootstrap_results.csv"))


if __name__ == "__main__":
    # Test ebcl random finetuning
    # python -m src.train.ebcl.bootstrap name=test_ebcl_bootstrap max_epochs=2 batch_size=256 dataloader_config.num_cpus=4
    os.environ['HYDRA_FULL_ERROR'] = "1"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    ebcl_finetune_app()
