import logging
import os
from functools import partial
import pandas as pd
import hydra
import pickle
import shutil

from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

from src.configs.train_configs import EncounterSet, save_config
from src.model.strats_model import StratsModule
from src.train.strats.strats_finetune import StratsFinetuneConfig, finetune_strats
from src.train.ebcl.ebcl_train_pl import get_trainer, EBCLLightningDataset
from src.data.dataloaders.cached_dataset import CachedDataLoaderCreator
from src.configs.dataloader_configs import SupervisedView


@hydra.main(version_base=None, config_name="strats_finetune_config")
def strats_multi_seed(cfg: StratsFinetuneConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"
    logging.info("Setting up config...")
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    assert cfg.dataloader_config.encounter_set in [
        EncounterSet.MORTALITY,
        EncounterSet.READMISSION,
        EncounterSet.LOS,
    ]  # should run on mortality or readmission or los

    if cfg.dataloader_config.encounter_set == EncounterSet.LOS:
        assert cfg.dataloader_config.modalities == SupervisedView.PRE
    assert isinstance(cfg, StratsFinetuneConfig)
    assert not cfg.debug
    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")

    # Pull lr and dropout from best model
    cfg.seed = None
    output_path = cfg.get_output_path()

    with open(os.path.join(output_path, "ray_strats_finetune_tune_result.pkl"), 'rb') as file:
        df = pickle.load(file)
    best_model = df.sort_values(by='val_loss').iloc[0]
    cfg.model_config.lr = float(best_model['config/train_loop_config/lr'])
    cfg.model_config.dropout = float(best_model['config/train_loop_config/dropout'])
    cfg.seed = int(best_model['config/train_loop_config/seed'])

    cfg.logger_config.save_dir = '/storage/x-x-x/results/strats_finetune/multiseed'

    # Run muiltiseed finetuning with best model
    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")

    hparam_config = {
        "lr": cfg.model_config.lr,
        "dropout": cfg.model_config.dropout,
        "seed": tune.grid_search([cfg.seed, 2000, 1991, 926, 1006]), 
    }

    scaling_config = ScalingConfig(
        num_workers=1,
        use_gpu=True,
        resources_per_worker={"CPU": 2, "GPU": 1.0},
    )
    run_config = RunConfig(
        storage_path=cfg.get_output_path(),
        name=cfg.logger_config.name,
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_loss",
            checkpoint_score_order="min",
        ),
    )
    ray_trainer = TorchTrainer(
        partial(finetune_strats, cfg, True),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            num_samples=1,  # number of hyperparameter samples to try
        ),
        param_space={"train_loop_config": hparam_config},
    )
    results = tuner.fit()

    test_results = []

    ebcl_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = EBCLLightningDataset(ebcl_dataloader_creator)
    dm.setup()
    for i, result in enumerate(results):
        result_checkpoint = result.checkpoint.path + "/checkpoint.ckpt"
        checkpoint_dir = output_path + f"/checkpoint_{i}/checkpoint.ckpt"
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        shutil.copy(result_checkpoint, checkpoint_dir)
        ocp_model = StratsModule.load_from_checkpoint(cfg=cfg.model_config, checkpoint_path=checkpoint_dir)
        assert not cfg.half_dtype
        print(f"Sweet, you can load the model from: {checkpoint_dir}")
        trainer = get_trainer(cfg, use_ray=False)
        test_result = trainer.test(model=ocp_model, datamodule=dm, ckpt_path=checkpoint_dir)
        test_results.extend(test_result)
    pd.DataFrame(test_results).to_csv(os.path.join(output_path, f"test_results.csv"))


if __name__ == "__main__":
    # python -m src.train.strats.multiseed_ray dataloader_config.encounter_set=MORTALITY pretrain_ckpt='/storage/x-x-x/server_results/strats_pretrain/pretrain_medium_pre_and_post_admissionstrats/best_checkpoint/checkpoint.ckpt' pretrain_sampling=ADMISSION max_epochs=1
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN)
    strats_multi_seed()