import os
import hydra
import pickle
import random
import logging
import numpy as np

from src.train.ebcl.ebcl_finetune import ebcl_finetune_app, EbclFinetuneConfig
from src.configs.train_configs import EncounterSet


def do_bootstrap(self, preds, pred_vals, trues, n=1000):
    auc_list = []
    apr_list = []
    acc_list = []
    f1_list = []

    rng = np.random.RandomState(seed=1)
    for _ in range(n):
        idxs = rng.choice(len(trues), size=len(trues), replace=True)
        if len(set(trues[idxs])) < 2:
            continue
        pred_arr= preds[idxs]
        true_arr = trues[idxs]
        pred_val_arr = pred_vals[idxs]

        auc = roc_auc_score(true_arr, pred_arr)
        apr = average_precision_score(true_arr, pred_arr)
        acc = accuracy_score(true_arr, pred_val_arr)
        f1 = f1_score(true_arr, pred_val_arr)
        
        auc_list.append(auc)
        apr_list.append(apr)
        acc_list.append(acc)
        f1_list.append(f1)

    return np.array(auc_list), np.array(apr_list), np.array(acc_list), np.array(f1_list)

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

    for i in range(5):
        if i == 0:
            cfg.seed = int(best_model['config/train_loop_config/seed'])
        else:
            cfg.seed = random.randint(2, 10000)
        ebcl_finetune_app(cfg)

    return df, cfg

if __name__ == "__main__":
    # python -m src.train.test_finetuned_models dataloader_config.encounter_set=MORTALITY pretrain_ckpt='/storage/x-x-x/server_results/ebcl_pretrain/pretrain_medium_pre_and_post_admissionebcl/best_checkpoint/checkpoint.ckpt' pretrain_sampling=ADMISSION
    # python -m src.train.test_finetuned_models dataloader_config.num_cpus=$NUM_CPUS half_dtype=$HALF_DTYPE pretrain_ckpt=$CKPT dataloader_config.encounter_set=$ENCOUNTER_SET pretrain_sampling=$PRETRAIN gpu_ids=$GPU_IDS
    os.environ['HYDRA_FULL_ERROR'] = "1"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    get_best_result_app()
