import logging
import os
import pickle
from functools import partial
from typing import Optional, Union

import hydra
import pandas as pd
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from src.configs.dataloader_configs import (
    BaseDataloaderConfig,
    CacheMode,
    DuettDataloaderConfig,
    EBCLDataloaderConfig,
    EBCLSamplingMethod,
    SupervisedView,
)
from src.configs.utils import hydra_dataclass
from src.data.dataloaders.duett_dataloader import (
    DuettDataLoaderCreator,
    DuettDataset,
    collate_duett_triplets,
)
from src.data.dataloaders.ebcl_dataloader import (
    EBCLDataLoaderCreator,
    EBCLDataset,
    collate_triplets,
)
from src.data.dataloaders.utils import BaseDataloaderCreator
from src.utils.data_utils import EncounterSet, Split


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: Optional[Union[DuettDataset, EBCLDataset, str]] = None):
        if isinstance(dataset, str):
            with open(dataset, "rb") as f:
                try:
                    self.cache = pickle.load(f)
                except ModuleNotFoundError:
                    self.cache = pd.compat.pickle_compat.load(f)
                    print("Loading data via pd.compat!")
        else:
            assert isinstance(dataset, torch.utils.data.Dataset)
            self.dataset = dataset
            self.cache = {}
            # if isinstance(dataset, DuettDataset):
            #     assert dataset.args.num_events == 32, "num_events should be 32"
            dataloader = DataLoader(
                dataset,
                batch_size=dataset.args.batch_size,
                shuffle=False,
                num_workers=dataset.args.num_cpus,
                collate_fn=lambda x: x,
            )
            i = 0
            for batch in tqdm(dataloader):
                for sample in batch:
                    self.cache[i] = self._deidentify_sample(sample)
                    i += 1

    def _deidentify_sample(self, sample):
        for key in list(sample.keys()):
            if key not in self.dataset.deidentified_keys:
                del sample[key]
        return sample

    def save(self, cache_file):
        with open(cache_file, "wb") as f:
            pickle.dump(self.cache, f)

    def __getitem__(self, index):
        return self.cache[index]

    def __len__(self):
        return len(self.cache)


class CachedDataLoaderCreator(BaseDataloaderCreator):
    def __init__(self, args):
        self.args = args
        # Set collate function and cached file prefix
        if isinstance(self.args, EBCLDataloaderConfig):
            self.collate_func = partial(
                collate_triplets, self.args.modalities, self.args.half_dtype
            )
            self.prefix = "EBCL"
        elif isinstance(self.args, DuettDataloaderConfig):
            self.collate_func = partial(collate_duett_triplets, self.args.half_dtype)
            self.prefix = "DUETT"
        else:
            raise NotImplementedError(
                f"Unsupported dataloader config type: {type(self.args)}"
            )
        # Confirm cache_dir exists:
        os.makedirs(self.args.cache_dir, exist_ok=True)

    def get_file_path(self, split: Split):
        encounter_set: EncounterSet = self.args.encounter_set
        if self.args.modalities != SupervisedView.PRE_AND_POST:
            suffix = f"_{self.args.modalities.value}"
        else:
            suffix = ""
        if isinstance(self.args, DuettDataloaderConfig):
            return os.path.join(
                self.args.cache_dir,
                f"{self.prefix}_{encounter_set.value}_{split.value}{suffix}_nevents{self.args.num_events}.pkl",
            )
        else:
            return os.path.join(
                self.args.cache_dir,
                f"{self.prefix}_{encounter_set.value}_{split.value}{suffix}.pkl",
            )

    def save_datasets(
        self,
        train_dataset: torch.utils.data.Dataset,
        val_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
    ):
        assert (
            self.args.cache_mode == CacheMode.CREATE_CACHE
        ), "cache_mode should be CREATE_CACHE"
        CachedDataset(train_dataset).save(self.get_file_path(Split.TRAIN))
        CachedDataset(val_dataset).save(self.get_file_path(Split.VAL))
        CachedDataset(test_dataset).save(self.get_file_path(Split.TEST))

    def get_dataset(self, split):
        assert self.args.subsample == 1, "subsample should be 1"
        dataset = CachedDataset(self.get_file_path(split))
        return dataset

    def _get_dataloader(self, dataset, split):
        drop_last = split != Split.TEST
        shuffle = split != Split.TEST
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.args.num_cpus,
            pin_memory=True,
            collate_fn=self.collate_func,
        )

    def get_dataloaders(self):
        assert (
            self.args.cache_mode == CacheMode.LOAD_CACHE
        ), "cache_mode should be LOAD_CACHE"
        train_dataset = self.get_dataset(Split.TRAIN)
        test_dataset = self.get_dataset(Split.TEST)
        val_dataset = self.get_dataset(Split.VAL)

        train_dataloader = self._get_dataloader(train_dataset, split=Split.TRAIN)
        test_dataloader = self._get_dataloader(test_dataset, split=Split.TEST)
        val_dataloader = self._get_dataloader(val_dataset, split=Split.VAL)
        return train_dataloader, val_dataloader, test_dataloader


def cache_datasets(
    args: BaseDataloaderConfig, dataloader_creator: BaseDataloaderCreator
):
    train_dataset = dataloader_creator.get_dataset(args.encounter_set, Split.TRAIN)
    val_dataset = dataloader_creator.get_dataset(args.encounter_set, Split.VAL)
    test_dataset = dataloader_creator.get_dataset(args.encounter_set, Split.TEST)
    cache_dataloader_creator = CachedDataLoaderCreator(args)
    cache_dataloader_creator.save_datasets(
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset
    )


@hydra_dataclass
class CacheConfig:
    cache_dir: str = MISSING
    num_cpus: int = 8
    duett_dl_config: DuettDataloaderConfig = DuettDataloaderConfig(
        cache_mode=CacheMode.CREATE_CACHE
    )
    ebcl_dl_config: EBCLDataloaderConfig = EBCLDataloaderConfig(
        cache_mode=CacheMode.CREATE_CACHE,
        sampling_method=EBCLSamplingMethod.ADMISSION,
        modalities=SupervisedView.PRE_AND_POST,
    )

    def __post_init__(self):
        # set cache_dir for duett_dl_config and ebcl_dl_config
        self.duett_dl_config.cache_dir = self.cache_dir
        self.ebcl_dl_config.cache_dir = self.cache_dir
        self.duett_dl_config.num_cpus = self.num_cpus
        self.ebcl_dl_config.num_cpus = self.num_cpus


cs = ConfigStore.instance()
cs.store(name="cache_config", node=CacheConfig)


@hydra.main(version_base=None, config_name="cache_config")
def cache_app(cfg: CacheConfig) -> None:
    assert (
        cfg.duett_dl_config.cache_mode == CacheMode.CREATE_CACHE
    ), "cache_mode should be CREATE_CACHE"
    assert (
        cfg.ebcl_dl_config.cache_mode == CacheMode.CREATE_CACHE
    ), "cache_mode should be CREATE_CACHE"
    encounter_sets = [
        EncounterSet.READMISSION,
        EncounterSet.MORTALITY,
        EncounterSet.SUFFICIENT,
        EncounterSet.LOS,
    ]
    # cache duett datasets
    logging.info("Step 1: Caching DUETT Datasets")
    dl_cfg = cfg.duett_dl_config
    logging.info("Step 2: Caching EBCL Datasets")
    dl_cfg = cfg.ebcl_dl_config
    for modality in SupervisedView:
        for encounter_set in encounter_sets:
            if encounter_set == EncounterSet.LOS and modality != SupervisedView.PRE:
                continue
            if modality != SupervisedView.EARLY_FUSION:
                continue
            logging.info(
                f"Caching EBCL Dataset for encounter_set, modality: {encounter_set.value}, {modality.value}"
            )
            dl_cfg.modalities = modality
            dl_cfg = cfg.ebcl_dl_config
            dl_cfg.cache_dir = cfg.cache_dir
            dl_cfg.num_cpus = cfg.num_cpus
            dl_cfg.encounter_set = encounter_set
            dl_cfg_object = hydra.utils.instantiate(dl_cfg, _convert_="object")
            cache_datasets(dl_cfg_object, EBCLDataLoaderCreator(dl_cfg_object))


if __name__ == "__main__":
    # python -m src.data.dataloaders.cached_dataset cache_dir=x-x-xhf_cohort/final/cache/ num_cpus=16 duett_dl_config.num_events=128 duett_dl_config.store_data=True
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    cache_app()
