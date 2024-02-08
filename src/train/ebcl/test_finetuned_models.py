import os
import pickle
import random
import logging
import hydra
import pandas as pd

from src.train.ebcl.ebcl_finetune import ebcl_finetune_app, EbclFinetuneConfig
from src.configs.train_configs import EncounterSet
from src.model.ebcl_model import EBCLModule
from src.train.ebcl.ebcl_train_pl import get_trainer, EBCLLightningDataset
from src.data.dataloaders.cached_dataset import CachedDataLoaderCreator

@hydra.main(version_base=None, config_name="ebcl_finetune_config")
def get_best_result_app(cfg: EbclFinetuneConfig):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logging.info("Setting up config...")
    cfg = hydra.utils.instantiate(cfg, _convert_="object")
    assert cfg.model_config.pretrain is False
    assert cfg.dataloader_config.encounter_set in [
        EncounterSet.READMISSION,
        EncounterSet.MORTALITY,
        EncounterSet.LOS
    ]  # should run on whole dataset

    cfg.seed = None
    output_path = cfg.get_output_path()

    with open(os.path.join(output_path, "ray_ebcl_finetune_tune_result.pkl"), 'rb') as file:
        df = pickle.load(file)
    best_model = df.sort_values(by='val_loss').iloc[0]
    cfg.model_config.lr = float(best_model['config/train_loop_config/lr'])
    cfg.model_config.dropout = float(best_model['config/train_loop_config/dropout'])

    cfg.logger_config.save_dir = '/storage/x-x-xx/results/ebcl_finetune/multiseed'

    for i in range(5):
        if i == 0:
            cfg.seed = int(best_model['config/train_loop_config/seed'])
        else:
            cfg.seed = random.randint(2, 10000)
        results = ebcl_finetune_app(cfg)
        output_path = cfg.get_output_path()
        pd.DataFrame([results]).to_csv(os.path.join(output_path, f"test_results_seed_{cfg.seed}.csv"))
        print(results)

    return df, cfg


if __name__ == "__main__":
    # python -m src.train.test_finetuned_models dataloader_config.encounter_set=MORTALITY pretrain_ckpt='/storage/x-x-x/server_results/ebcl_pretrain/pretrain_medium_pre_and_post_admissionebcl/best_checkpoint/checkpoint.ckpt' pretrain_sampling=ADMISSION max_epochs = 2
    # python -m src.train.test_finetuned_models dataloader_config.num_cpus=$NUM_CPUS half_dtype=$HALF_DTYPE pretrain_ckpt=$CKPT dataloader_config.encounter_set=$ENCOUNTER_SET pretrain_sampling=$PRETRAIN gpu_ids=$GPU_IDS
    os.environ['HYDRA_FULL_ERROR'] = "1"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    get_best_result_app()