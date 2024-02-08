import logging
import os
from functools import partial

import hydra
from ray import tune
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from src.configs.train_configs import (
    EncounterSet,
    PretrainMethod,
    save_config,
)
from src.model.duett_model_static import DuettModule, load_fine_tune_duett
from src.train.duett.duett_train_pl import DuettPretrainTuneConfig, train_duett


@hydra.main(version_base=None, config_name="duett_pretrain_tune_config")
def duett_ray_app(cfg: DuettPretrainTuneConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    os.environ["RAY_memory_monitor_refresh_ms"] = "0"  # should stop the OOM error
    logging.info("Setting up config...")
    assert cfg.pretrain_method == PretrainMethod.DUETT
    assert (
        cfg.dataloader_config.encounter_set == EncounterSet.SUFFICIENT
    )  # should run on whole dataset
    cfg: DuettPretrainTuneConfig = hydra.utils.instantiate(cfg, _convert_="object")
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
        partial(train_duett, cfg, True),
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
    _ = load_fine_tune_duett(cfg,  best_checkpoint_dir + "/checkpoint.ckpt")
    print(f"Sweet, you can load the model from: {best_checkpoint_dir}/checkpoint.ckpt")


if __name__ == "__main__":
    # python -m src.train.duett.duett_ray dataloader_config.num_cpus=8 dataloader_config.num_events=128 model_config.num_event=128
    # import ray
    # gpu_ids = [3, 4, 5, 6, 7]
    # ray.init(gpu_ids=gpu_ids)
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN)
    duett_ray_app()
