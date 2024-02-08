import os
import pickle
import random
import logging
from src.train.ocp.ocp_finetune import ocp_finetune_app, OcpFinetuneConfig
import hydra
from src.configs.train_configs import EncounterSet


@hydra.main(version_base=None, config_name="ocp_finetune_config")
def get_best_result_app(cfg: OcpFinetuneConfig):
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
    
    with open(os.path.join(output_path, "ray_ocp_finetune_tune_result.pkl"), 'rb') as file:
        df = pickle.load(file)
    best_model = df.sort_values(by='val_loss').iloc[0]
    cfg.model_config.lr = float(best_model['config/train_loop_config/lr'])
    cfg.model_config.dropout = float(best_model['config/train_loop_config/dropout'])

    for i in range(5):
        if i == 0:
            cfg.seed = int(best_model['config/train_loop_config/seed'])
        else:
            cfg.seed = random.randint(2, 10000)
        ocp_finetune_app(cfg)

    return df, cfg

if __name__ == "__main__":
    # python -m src.train.test_finetuned_models dataloader_config.encounter_set=MORTALITY pretrain_ckpt='/storage/x-x-x/server_results/ocp_pretrain/pretrain_medium_early_fusion/best_checkpoint/checkpoint.ckpt' pretrain_sampling=ADMISSION
    # python -m src.train.test_finetuned_models dataloader_config.num_cpus=$NUM_CPUS half_dtype=$HALF_DTYPE pretrain_ckpt=$CKPT dataloader_config.encounter_set=$ENCOUNTER_SET pretrain_sampling=$PRETRAIN gpu_ids=$GPU_IDS
    
    # Test strats debug finetuning
    # python -m src.train.ocp.ocp_finetune name=test_strats_finetune max_epochs=2 dataloader_config.num_cpus=4 half_dtype=True modalities=PRE_AND_POST dataloader_config.encounter_set=MORTALITY debug=True
    # Test strats actual finetuning
    # python -m src.train.ocp.ocp_finetune name=test_ebcl_finetune max_epochs=2 batch_size=256 dataloader_config.num_cpus=4 half_dtype=True modalities=PRE_AND_POST dataloader_config.encounter_set=MORTALITY

    os.environ['HYDRA_FULL_ERROR'] = "1"
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    get_best_result_app()
