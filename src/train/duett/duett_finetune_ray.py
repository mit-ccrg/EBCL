import logging
import os
from functools import partial

import pandas as pd
import hydra
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from src.configs.train_configs import EncounterSet, PretrainMethod, save_config
from src.model.duett_model_static import load_fine_tune_duett
from src.train.duett.duett_finetune import DuettFinetuneConfig, finetune_duett
from src.train.duett.duett_train_pl import get_trainer, DuettLightningDataset
from src.data.dataloaders.cached_dataset import CachedDataLoaderCreator
from src.configs.dataloader_configs import SupervisedView


@hydra.main(version_base=None, config_name="duett_finetune_config")
def duett_finetune_ray_app(cfg: DuettFinetuneConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # should stop the OOM error
    logging.info("Setting up config...")
    assert cfg.pretrain_method == PretrainMethod.DUETT

    assert cfg.dataloader_config.encounter_set in [
        EncounterSet.MORTALITY,
        EncounterSet.READMISSION,
        EncounterSet.LOS
    ]  # should run on mortality or readmission of LOS
    if cfg.dataloader_config.encounter_set == EncounterSet.LOS: # LOS is PRE only
        assert cfg.dataloader_config.modalities == SupervisedView.PRE
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    assert isinstance(cfg, DuettFinetuneConfig)
    output_path = cfg.get_output_path()
    os.makedirs(output_path, exist_ok=True)
    save_config(cfg, output_path + "/config.yml")

    hparam_config = {
        "lr": tune.loguniform(1e-6, 1e-2),
        "dropout": tune.uniform(0, 0.6),
        "seed": tune.randint(2, 10000),
    }

    scheduler = ASHAScheduler(
        time_attr="epoch",
        metric="val_loss",
        mode="min",
        max_t=cfg.max_epochs,  # max epochs to do
        grace_period=cfg.grace_period,  # do successive halving every grace_period epochs
        reduction_factor=2,
    )

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
        partial(finetune_duett, cfg, True),
        scaling_config=scaling_config,
        run_config=run_config,
    )
    tuner = tune.Tuner(
        ray_trainer,
        tune_config=tune.TuneConfig(
            num_samples=cfg.num_samples,  # number of hyperparameter samples to try
            scheduler=scheduler,
        ),
        param_space={"train_loop_config": hparam_config},
    )
    results = tuner.fit()
    result_df_path = os.path.join(output_path, f"{cfg.name}_tune_result.pkl")
    results.get_dataframe().to_pickle(result_df_path)
    best_trial = results.get_best_result("val_loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['val_loss']}")
    best_checkpoint_dir = best_trial.checkpoint.to_directory(
        path=output_path + "/best_checkpoint"
    )
    print(f"Best checkpoint dir: {best_checkpoint_dir}")
    duett_model = load_fine_tune_duett(cfg.model_config, best_checkpoint_dir + "/checkpoint.ckpt")
    print(f"Sweet, you can load the model from: {best_checkpoint_dir}/checkpoint.ckpt")
    trainer = get_trainer(cfg, use_ray=False)
    duett_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = DuettLightningDataset(duett_dataloader_creator)
    dm.setup()
    results = trainer.test(model=duett_model, datamodule=dm, ckpt_path=best_checkpoint_dir + "/checkpoint.ckpt")[0]
    pd.DataFrame([results]).to_csv(os.path.join(output_path, f"test_results_seed_{best_trial.config['train_loop_config']['seed']}.csv"))


if __name__ == "__main__":
    # python -m src.train.duett.duett_finetune_ray dataloader_config.encounter_set=MORTALITY logger_config.name=mortality_finetune_duett_ray
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN)
    duett_finetune_ray_app()
