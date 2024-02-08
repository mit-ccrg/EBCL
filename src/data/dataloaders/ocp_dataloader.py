import logging
import os
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch

from src.configs.dataloader_configs import OCPDataloaderConfig, SupervisedView
from src.data.dataloaders.utils import (
    L2_KEYS,
    L3_KEYS,
    BaseDataloaderCreator,
    collate_value,
    load_encounters,
    load_mrns,
    process_df_to_dict,
)
from src.utils.data_utils import Split


def collate_ocp_triplets(half_dtype, batch, device=None):
    collated_batch = dict(
        early_fusion=dict(
            cat=dict(
                date=None,
                variable=None,
                value=None,
            ),
            cont=dict(
                date=None,
                variable=None,
                value=None,
            ),
        )
    )

    L1_KEYS = [SupervisedView.EARLY_FUSION.value]
    lengths = []

    for l1 in L1_KEYS:
        for l2 in L2_KEYS:
            for l3 in L3_KEYS:
                # import pdb; pdb.set_trace() # [len(data["early_fusion"]["cat"]["date"]) for data in batch]
                tensors, lengths = collate_value(batch, half_dtype, l1, l2, l3, device)
                tensors.to(device)
                collated_batch[l1][l2][l3] = tensors
            collated_batch[l1][l2]["length"] = lengths

    mrns = [each["hospital_mrn"] for each in batch]
    flip = [each["flip"] for each in batch]
    collated_batch["hospital_mrn"] = mrns
    collated_batch["flip"] = torch.as_tensor(flip, device=device)

    encounter_date = [each["encounter_date"] for each in batch]
    collated_batch["encounter_date"] = encounter_date
    collated_batch["early_fusion"]["cont"]["value"] = collated_batch["early_fusion"]["cont"][
        "value"
    ].clip(-100, 100)

    return collated_batch


class OCPDataset(torch.utils.data.Dataset):
    def __init__(self, args: OCPDataloaderConfig, data_dict, encounters, split: Split):
        self.args = args
        self.data_dict = data_dict
        self.split = split
        split_mrns = load_mrns(args, split)
        self.mrns = []

        for mrn in split_mrns:
            if len(self.data_dict[mrn]) >= 2 * args.min_obs:
                self.mrns.append(mrn)
        self.mrns = np.array(self.mrns)
        # shuffle the mrns using data_seed
        np.random.default_rng(seed=args.seed).shuffle(self.mrns)

        self.std_time = encounters.date.std()

    def generate_ocp_data(self, idx, swap):
        """
        left_cutoff: used to make left cutoff of dataset of args.max_obs size dataset
        right_cutoff: used to make right cutoff of dataset of args.max_obs size dataset
        """
        args = self.args
        std_time = self.std_time
        hospital_mrn = self.mrns[idx]
        subset_df = self.data_dict[hospital_mrn].copy()
        assert subset_df.date.is_monotonic_increasing

        assert (
            len(subset_df) >= 2 * args.min_obs
        ), f"ocp sample has only {len(subset_df)} samples!"

        if len(subset_df) <= args.max_obs:
            left = 0
        else:
            if self.split == Split.TRAIN:
                left = np.random.randint(low=0, high=len(subset_df) - args.max_obs)
            else:
                # use constant seed for validation and test set
                constant_rng = np.random.default_rng(seed=idx)
                left = constant_rng.integers(low=0, high=len(subset_df) - args.max_obs)

        # Sequence must be even length, so pre and post are the same size
        right = min(left + args.max_obs, left + len(subset_df) - len(subset_df) % 2)
        cutoff = int((left + right) / 2)

        pre_filtered_rows = subset_df.iloc[left:cutoff].reset_index(drop=True)
        post_filtered_rows = subset_df.iloc[cutoff:right].reset_index(drop=True)

        gap = post_filtered_rows["date"].min() - pre_filtered_rows["date"].max()
        if swap:
            date_add = (post_filtered_rows["date"].max() - pre_filtered_rows["date"].min())
            pre_filtered_rows['date'] += date_add + gap
            enc_date = post_filtered_rows["date"].max()  # Post max
            filtered_rows = pd.concat([post_filtered_rows, pre_filtered_rows], axis=0)
            assert enc_date <= pre_filtered_rows["date"].min()
        else:
            filtered_rows = pd.concat([pre_filtered_rows, post_filtered_rows], axis=0, ignore_index=True)
            enc_date = pre_filtered_rows["date"].max()  # Pre max
            assert enc_date <= post_filtered_rows["date"].min()
        filtered_rows = filtered_rows.sort_values(by="date").reset_index(drop=True)
        assert filtered_rows.date.is_monotonic_increasing

        obs = dict(
            hospital_mrn=hospital_mrn,
            encounter_date=enc_date,
            flip=swap,
        )
        obs[SupervisedView.EARLY_FUSION.value] = process_df_to_dict(filtered_rows, enc_date, std_time)

        return obs

    def get_encounter(self, idx, swap):
        return self.generate_ocp_data(idx, swap)

    def __getitem__(self, idx):
        # swap: label whether the sequence is swapped
        if self.split == Split.TRAIN:
            swap = np.random.choice([0, 1])
        else:
            # use constant seed for validation and test set
            swap = idx % 2

        if isinstance(idx, int):
            return self.get_encounter(idx, swap)
        else:
            raise NotImplementedError("Only int indexing is supported")
            return [self.get_encounter(i, swap) for i in idx]

    def __len__(self):
        return len(self.mrns)


class OCPDataLoaderCreator(BaseDataloaderCreator):
    def __init__(self, args: OCPDataloaderConfig):
        self.args = args
        logging.info("Loading Timeseries Data")

        with open(os.path.join(args.data_dir, "timeseries_dict.pkl"), "rb") as f:
            try:
                self.data_dict = pickle.load(f)
            except ModuleNotFoundError:
                self.data_dict = pd.compat.pickle_compat.load(f)
                print("Loading data via pd.compat!")

    def get_dataset(self, encounter_set, split):
        frac = 1
        encounters = load_encounters(
            self.args, encounter_set, split, self.args.inpatient_only
        ).sample(frac=frac, random_state=self.args.seed)
        logging.info(f"len(encounters) for split {split.value}: {len(encounters)}")
        dataset = OCPDataset(self.args, self.data_dict, encounters, split)

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
            collate_fn=partial(collate_ocp_triplets, self.args.half_dtype),
        )


if __name__ == "__main__":
    # python -m src.data.dataloaders.ocp_dataloader
    import shutil
    import tempfile
    import torch

    from src.utils.data_utils import EncounterSet

    from tests.test_dataloaders import generate_synthetic_data

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for tests: {temp_dir}")

    generate_synthetic_data(temp_dir)

    args = OCPDataloaderConfig(encounter_set=EncounterSet.MORTALITY, data_dir=temp_dir)
    # data_dict = pd.compat.pickle_compat.load(open(os.path.join("/storage/shared/hf_cohort/final/", "timeseries_dict.pkl"), "rb")) 
    data_dict = pickle.load(open(os.path.join(temp_dir, "timeseries_dict.pkl"), "rb"))

    encounters = load_encounters(
        args,
        encounter_set=EncounterSet.MORTALITY,
        split=Split.TRAIN,
        inpatient_only=True,
    )
    dataset = OCPDataset(args, data_dict, encounters, Split.TRAIN)
    output = dataset.generate_ocp_data(0, swap=True)
    assert isinstance(output, dict)
    output = dataset.generate_ocp_data(0, swap=False)
    assert isinstance(output, dict)
    # Cleanup after all tests have run
    shutil.rmtree(temp_dir)
    print(f"Deleted temporary directory for tests: {temp_dir}")