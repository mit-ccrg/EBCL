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
from src.model.ocp_model import OCPModule
from src.train.ocp.ocp_train_pl import OcpPretrainTuneConfig, train_ocp


@hydra.main(version_base=None, config_name="ocp_pretrain_tune_config")
def ocp_ray_app(cfg: OcpPretrainTuneConfig):
    logging.info("Setting up config...")
    assert (
        cfg.dataloader_config.encounter_set == EncounterSet.SUFFICIENT
    )  # should run on whole dataset
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    output_path = cfg.get_output_path()
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
        partial(train_ocp, cfg, True),
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
    _ = OCPModule.load_from_checkpoint(
        cfg=cfg.model_config,
        checkpoint_path=best_checkpoint_dir + "/checkpoint.ckpt",
    )
    print(f"Sweet, you can load the model from: {best_checkpoint_dir}/checkpoint.ckpt")


if __name__ == "__main__":
    # python -m src.train.ocp.ocp_ray dataloader_config.num_cpus=8
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN)
    ocp_ray_app()
