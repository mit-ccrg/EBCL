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
from src.model.ocp_model import OCPModule
from src.train.ocp.ocp_finetune import OcpFinetuneConfig, finetune_ocp
from src.train.ebcl.ebcl_train_pl import get_trainer, EBCLLightningDataset
from src.data.dataloaders.cached_dataset import CachedDataLoaderCreator


@hydra.main(version_base=None, config_name="ocp_finetune_config")
def ocp_finetune_ray_app(cfg: OcpFinetuneConfig):
    # os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logging.info("Setting up config...")
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    assert cfg.dataloader_config.encounter_set in [
        EncounterSet.MORTALITY,
        EncounterSet.READMISSION,
        EncounterSet.LOS,
    ]  # should run on mortality or readmission
    assert isinstance(cfg, OcpFinetuneConfig)
    assert not cfg.debug
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
        resources_per_worker={"CPU": 2, "GPU": 1},
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
        partial(finetune_ocp, cfg, True),
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
    ocp_model = OCPModule.load_from_checkpoint(cfg=cfg.model_config, checkpoint_path=best_checkpoint_dir + "/checkpoint.ckpt")
    if cfg.model_config.half_dtype:
        print(f"Using Half Precision ->")
        ocp_model.half()
    print(f"Sweet, you can load the model from: {best_checkpoint_dir}/checkpoint.ckpt")
    trainer = get_trainer(cfg, use_ray=False)
    ebcl_dataloader_creator = CachedDataLoaderCreator(cfg.dataloader_config)
    dm = EBCLLightningDataset(ebcl_dataloader_creator)
    dm.setup()
    results = trainer.test(model=ocp_model, datamodule=dm, ckpt_path=best_checkpoint_dir + "/checkpoint.ckpt")[0]
    pd.DataFrame([results]).to_csv(os.path.join(output_path, f"test_results_seed_{best_trial.config['train_loop_config']['seed']}.csv"))


if __name__ == "__main__":
    # python -m src.train.ocp.ocp_finetune_ray dataloader_config.encounter_set=MORTALITY
    logging.basicConfig()
    logging.getLogger().setLevel(logging.WARN)
    ocp_finetune_ray_app()
