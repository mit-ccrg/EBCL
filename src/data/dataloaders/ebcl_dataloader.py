import logging
import os
import pickle
from functools import partial

import torch
import pandas as pd
import numpy as np

from src.configs.dataloader_configs import EBCLDataloaderConfig, SupervisedView, EBCLSamplingMethod
from src.data.dataloaders.utils import (
    L2_KEYS,
    L3_KEYS,
    BaseDataloaderCreator,
    collate_value,
    load_encounters,
    load_ablation_encounters,
    load_censored_encounters,
    process_df_to_dict,
)
from src.utils.data_utils import Split, EncounterSet


def collate_triplets(modalities, half_dtype, batch, device=None):
    collated_batch = dict(
        pre=dict(
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
        ),
        post=dict(
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
        ),
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
        ),
    )

    lengths = []

    if modalities == SupervisedView.EARLY_FUSION:
        L1_KEYS = [SupervisedView.EARLY_FUSION.value]
        del collated_batch["post"]
        del collated_batch["pre"]
    elif modalities == SupervisedView.PRE:
        L1_KEYS = ["pre"]
        del collated_batch["early_fusion"], collated_batch["post"]
    elif modalities == SupervisedView.POST:
        L1_KEYS = ["post"]
        del collated_batch["early_fusion"], collated_batch["pre"]
    else:
        L1_KEYS = ["pre", "post"]
        del collated_batch["early_fusion"]

    for l1 in L1_KEYS:
        for l2 in L2_KEYS:
            for l3 in L3_KEYS:
                tensors, lengths = collate_value(batch, half_dtype, l1, l2, l3, device)
                if not tensors.isfinite().all():
                    assert l3 == 'value' and l2 == 'cont', f"l3: {l3}, l2: {l2}"
                tensors.to(device)
                collated_batch[l1][l2][l3] = tensors
            collated_batch[l1][l2]["length"] = lengths
    try:
        mrns = [each["hospital_mrn"] for each in batch]
        collated_batch["hospital_mrn"] = mrns

        encounter_date = [each["encounter_date"] for each in batch]
        collated_batch["encounter_date"] = encounter_date

        encounter_type = [each["type"] for each in batch]
        collated_batch["type"] = encounter_type
    except:
        pass

    outcome = [each["outcome"] for each in batch]
    raw_outcome = [each["raw_outcome"] for each in batch]
    if outcome[0] is None:
        collated_batch["outcome"] = None
        collated_batch["raw_outcome"] = None
    else:
        collated_batch["outcome"] = torch.as_tensor(outcome, device=device)
        collated_batch["raw_outcome"] = torch.as_tensor(raw_outcome, device=device)

    # Clip continuous values
    for key in ["pre", "post", SupervisedView.EARLY_FUSION.value]:
        if key in collated_batch:
            collated_batch[key]["cont"]["value"] = collated_batch[key]["cont"][
                "value"
            ].clip(-100, 100)

    return collated_batch


MRN_TO_ENCOUNTER_SAMPLING_METHODS = [
    EBCLSamplingMethod.ADMISSION_PATIENT_LEVEL,
    EBCLSamplingMethod.ADMISSION_RANDOM_SAMPLE,
    EBCLSamplingMethod.CENSORED,
    EBCLSamplingMethod.OUTPATIENT,
    EBCLSamplingMethod.OUTPATIENT_AND_ED,
    EBCLSamplingMethod.OUTPATIENT_AND_ED_AND_UNKNOWN,
    EBCLSamplingMethod.INPATIENT_NO_CUTOFF,
    EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED,
    EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED_AND_UNKNOWN,
]

SAMPLING_METHODS_WITH_POST_LOS_CUTOFF = [
    EBCLSamplingMethod.ADMISSION,
    EBCLSamplingMethod.ADMISSION_PATIENT_LEVEL,
    EBCLSamplingMethod.ADMISSION_RANDOM_SAMPLE,
]

ABLATION_ENCOUNTERS = [
    EBCLSamplingMethod.OUTPATIENT,
    EBCLSamplingMethod.OUTPATIENT_AND_ED,
    EBCLSamplingMethod.OUTPATIENT_AND_ED_AND_UNKNOWN,
    EBCLSamplingMethod.INPATIENT_NO_CUTOFF,
    EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED,
    EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED_AND_UNKNOWN,
]


class EBCLDataset(torch.utils.data.Dataset):
    def __init__(self, args: EBCLDataloaderConfig, data_dict, encounters):
        self.args = args
        self.data_dict = data_dict
        self.encounters = encounters.reset_index(drop=True)
        if args.sampling_method == EBCLSamplingMethod.OUTPATIENT:
            assert set(self.encounters.type.unique()) == {"OUTPATIENT", "nod_OUTPATIENT"}
        elif args.sampling_method == EBCLSamplingMethod.OUTPATIENT_AND_ED:
            train_check = set(self.encounters.type.unique()) == {"OUTPATIENT", "nod_OUTPATIENT", "OUTPATIENT_EMERGENCY", "nod_OUTPATIENT_EMERGENCY"}
            test_and_val_check = set(self.encounters.type.unique()) == {"OUTPATIENT", "nod_OUTPATIENT", "OUTPATIENT_EMERGENCY"}
            assert train_check or test_and_val_check, set(self.encounters.type.unique())
        elif args.sampling_method == EBCLSamplingMethod.OUTPATIENT_AND_ED_AND_UNKNOWN:
            train_check = set(self.encounters.type.unique()) == {"OUTPATIENT", "nod_OUTPATIENT", "OUTPATIENT_EMERGENCY", "nod_OUTPATIENT_EMERGENCY", "UNKNOWN", "nod_UNKNOWN"}
            test_and_val_check = set(self.encounters.type.unique()) == {"OUTPATIENT", "nod_OUTPATIENT", "OUTPATIENT_EMERGENCY", "UNKNOWN", "nod_UNKNOWN"}
            assert train_check or test_and_val_check, set(self.encounters.type.unique())
        elif args.sampling_method == EBCLSamplingMethod.INPATIENT_NO_CUTOFF:
            assert set(self.encounters.type.unique()) == {"INPATIENT", "nod_INPATIENT"}
        elif args.sampling_method == EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED:
            train_check = set(self.encounters.type.unique()) == {"INPATIENT", "nod_INPATIENT", "OUTPATIENT_EMERGENCY", "nod_OUTPATIENT_EMERGENCY"}
            test_and_val_check = set(self.encounters.type.unique()) == {"INPATIENT", "nod_INPATIENT", "OUTPATIENT_EMERGENCY"}
            assert train_check or test_and_val_check, set(self.encounters.type.unique())
        elif args.sampling_method == EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED_AND_UNKNOWN:
            train_check = set(self.encounters.type.unique()) == {"INPATIENT", "nod_INPATIENT", "OUTPATIENT_EMERGENCY", "nod_OUTPATIENT_EMERGENCY", "UNKNOWN", "nod_UNKNOWN"}
            test_and_val_check = set(self.encounters.type.unique()) == {"INPATIENT", "nod_INPATIENT", "OUTPATIENT_EMERGENCY", "UNKNOWN", "nod_UNKNOWN"}
            assert train_check or test_and_val_check, set(self.encounters.type.unique())
        else:
            assert set(self.encounters.type.unique()) == {"INPATIENT"}

        self.mrn_list = list(self.encounters.hospital_mrn.unique())
        self.std_time = encounters.date.std()
        self.deidentified_keys = ["outcome", "raw_outcome", "pre", "post", "early_fusion"]
        if args.sampling_method in SAMPLING_METHODS_WITH_POST_LOS_CUTOFF:
            assert args.post_los_cutoff, "post_los_cutoff should be True"
        elif args.sampling_method in [EBCLSamplingMethod.OUTPATIENT, EBCLSamplingMethod.OUTPATIENT_AND_ED]:
            assert not args.post_los_cutoff, "post_los_cutoff should be False"
        if args.sampling_method in MRN_TO_ENCOUNTER_SAMPLING_METHODS:
            self.mrn_to_encounters = dict(list(self.encounters.groupby("hospital_mrn")))
        if args.ocp_and_strats_ablation:
            assert args.sampling_method == EBCLSamplingMethod.ADMISSION, "This is meant for generating embeddings on the finetuning dataset"

    def get_admission_pre_post(self, row, subset_df):
        args: EBCLDataloaderConfig = self.args
        # Sample all admissions per epoch, pre and post are the closest 512 to the admission.
        assert subset_df.date.is_monotonic_increasing
        pre_filtered_indexes = subset_df["date"] <= (row["date"])
        post_filtered_indexes = subset_df["date"] > (row["date"])
        if args.post_los_cutoff:
            post_filtered_indexes &= subset_df["date"] <= (
                row["date"] + row["los"] + pd.Timedelta(days=1)
            )
        if args.post_cutoff is not None:
            raise ValueError(
                "post_cutoff should be None, we are not doing this experiment now."
            )
            post_filtered_indexes &= subset_df["date"] <= (
                row["date"] + pd.Timedelta(days=args.post_cutoff)
            )
        if args.pre_cutoff is not None:
            raise ValueError(
                "pre_cutoff should be None, we are not doing this experiment now."
            )
            pre_filtered_indexes &= subset_df["date"] >= (
                row["date"] - pd.Timedelta(days=args.pre_cutoff)
            )
        encounter_date = row["date"]
        mrn = row["hospital_mrn"]
        pre_filtered_rows = subset_df[pre_filtered_indexes]
        post_filtered_rows = subset_df[post_filtered_indexes]
        return pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row


    def get_pre_and_post_df_indexes(self, idx):
        args: EBCLDataloaderConfig = self.args
        assert isinstance(args, EBCLDataloaderConfig), type(args)
        if self.args.sampling_method == EBCLSamplingMethod.RANDOM:
            # Sample a random event and take pre and post around that event, closest 512
            row = None
            mrn = self.mrn_list[idx]
            subset_df = self.data_dict[mrn]
            subset_df.reset_index(drop=True, inplace=True)
            assert subset_df.date.is_monotonic_increasing

            split_index = np.random.randint(low=self.args.min_obs, high=len(subset_df)+1 - self.args.min_obs)
            left = max(split_index - self.args.max_obs, 0)
            right = min(split_index + self.args.max_obs, len(subset_df))

            pre_filtered_rows = subset_df.iloc[left:split_index]
            post_filtered_rows = subset_df.iloc[split_index:right]
            encounter_date = pre_filtered_rows.date.max()
        elif self.args.sampling_method == EBCLSamplingMethod.RANDOM_WINDOW:
            # Sample two random events and take pre from first event and post from last event. Closest 512
            row = None
            mrn = self.mrn_list[idx]
            subset_df = self.data_dict[mrn]
            subset_df.reset_index(drop=True, inplace=True)
            assert subset_df.date.is_monotonic_increasing

            split_indexes = np.random.randint(size=2, low=self.args.min_obs, high=len(subset_df)+1 - self.args.min_obs)
            left_split_index = min(split_indexes)
            right_split_index = max(split_indexes)
            left = max(left_split_index - self.args.max_obs, 0)
            right = min(right_split_index + self.args.max_obs, len(subset_df))

            pre_filtered_rows = subset_df.iloc[left:left_split_index]
            post_filtered_rows = subset_df.iloc[right_split_index:right]
            encounter_date = pre_filtered_rows.date.max()
        elif self.args.sampling_method == EBCLSamplingMethod.ADMISSION_PATIENT_LEVEL:
            # Sample one admission per patient for the epoch. Closest 512 observations are taken.
            mrn = self.mrn_list[idx]
            row = self.mrn_to_encounters[mrn].sample(n=1).iloc[0]
            subset_df = self.data_dict[row["hospital_mrn"]]
            pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row = self.get_admission_pre_post(row, subset_df)
        elif self.args.sampling_method == EBCLSamplingMethod.CENSORED:
            # Sample one admission per patient for the epoch. 512 input that are preceding censoring window before admission (PRE) or following censoring window after admission.
            mrn = self.mrn_list[idx]
            row = self.mrn_to_encounters[mrn].sample(n=1).iloc[0]
            subset_df = self.data_dict[row["hospital_mrn"]]

            pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row = self.get_admission_pre_post(row, subset_df)

            assert pre_filtered_rows.date.is_monotonic_increasing
            assert post_filtered_rows.date.is_monotonic_increasing

            # censor the observations
            pre_filtered_rows = pre_filtered_rows[:-self.args.pre_obs_censor]
            post_filtered_rows = post_filtered_rows[self.args.post_obs_censor:]

        elif self.args.sampling_method == EBCLSamplingMethod.ADMISSION_RANDOM_SAMPLE:
            # Sample an admission per patient and take random pre and post around that event (random 512 before admission and in admission)
            mrn = self.mrn_list[idx]
            row = self.mrn_to_encounters[mrn].sample(n=1).iloc[0]
            subset_df = self.data_dict[row["hospital_mrn"]]
            pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row = self.get_admission_pre_post(row, subset_df)
        elif self.args.sampling_method in ABLATION_ENCOUNTERS:
            # Sample an outpatient admission
            mrn = self.mrn_list[idx]
            row = self.mrn_to_encounters[mrn].sample(n=1).iloc[0]
            subset_df = self.data_dict[row["hospital_mrn"]]
            pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row = self.get_admission_pre_post(row, subset_df)
        else:
            # Sample all admissions per epoch, pre and post are the closest 512 to the admission.
            assert self.args.sampling_method == EBCLSamplingMethod.ADMISSION
            row = self.encounters.iloc[idx]
            subset_df = self.data_dict[row["hospital_mrn"]]
            pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row = self.get_admission_pre_post(row, subset_df)
            
            assert min(pre_filtered_rows.shape[0], post_filtered_rows.shape[0]) >= args.min_obs
        
        return pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row

    def generate_ebcl_data_point(self, idx, return_df=False):
        """
        post_cutoff: used to make post_filtered indexes that span ~ [event, event+{post_cutoff}] days
        pre_cutoff: used to make post_filtered indexes that span ~ [event-{pre_cutoff}, event] days
        """
        pre_filtered_rows, post_filtered_rows, subset_df, mrn, encounter_date, row = self.get_pre_and_post_df_indexes(idx)
        args = self.args

        max_obs = args.max_obs
        if args.modalities == SupervisedView.EARLY_FUSION:
            assert args.sampling_method == EBCLSamplingMethod.ADMISSION, "This will actually work for any sampling method EXCEPT ADMISSION_RANDOM_SAMPLE"
            # For early fusion, we halve the max_obs because we are concatenating pre and post
            max_obs = max_obs // 2

        num_pre_samples = min(max_obs, len(pre_filtered_rows))
        num_post_samples = min(max_obs, len(post_filtered_rows))

        # get the most recent obsevations to the event if we have to remove datapoints
        if self.args.sampling_method == EBCLSamplingMethod.ADMISSION_RANDOM_SAMPLE:
            # Admission Random Sampling randomly samples 512 :)
            if len(pre_filtered_rows) >= max_obs:
                pre_filtered_rows.reset_index(inplace=True, drop=True)
                pre_filtered_rows = pre_filtered_rows.sample(num_pre_samples).sort_index()
            if len(post_filtered_rows) >= max_obs:
                post_filtered_rows.reset_index(inplace=True, drop=True)
                post_filtered_rows = post_filtered_rows.head(num_post_samples).sort_index()
        else:
            # All other sampling methods take the closest 512
            pre_filtered_rows = pre_filtered_rows.tail(num_pre_samples)
            post_filtered_rows = post_filtered_rows.head(num_post_samples)

        assert pre_filtered_rows.date.is_monotonic_increasing
        assert post_filtered_rows.date.is_monotonic_increasing

        if return_df:
            if row is None:
                return pre_filtered_rows, post_filtered_rows, encounter_date
            return pre_filtered_rows, post_filtered_rows, encounter_date, row['discharge_date']

        if args.modalities == SupervisedView.EARLY_FUSION:
            pre_filtered_rows = pd.concat([pre_filtered_rows, post_filtered_rows])
            obs = dict(
                hospital_mrn=mrn,
                encounter_date=encounter_date,
            )
            obs[SupervisedView.EARLY_FUSION.value] = process_df_to_dict(pre_filtered_rows, encounter_date, self.std_time)
        else:
            obs = dict(
                hospital_mrn=mrn,
                encounter_date=encounter_date,
            )
            if self.args.ocp_and_strats_ablation:
                pre_encounter_date = pre_filtered_rows.date.iloc[int(len(pre_filtered_rows) / 2)-1]
                post_encounter_date = post_filtered_rows.date.iloc[int(len(post_filtered_rows) / 2)-1]
            else:
                pre_encounter_date, post_encounter_date = encounter_date, encounter_date
            if args.modalities in [SupervisedView.PRE_AND_POST, SupervisedView.PRE]:
                obs['pre'] = process_df_to_dict(pre_filtered_rows, pre_encounter_date, self.std_time)

            if args.modalities in [SupervisedView.PRE_AND_POST, SupervisedView.POST]:
                obs['post'] = process_df_to_dict(post_filtered_rows, post_encounter_date, self.std_time)

        if row is not None:
            outcome = row.get("outcome")
            raw_outcome = row.get("raw_outcome")
            type = row["type"]
        else:
            outcome, raw_outcome, type = None, None, None
        obs["outcome"] = outcome
        obs["raw_outcome"] = raw_outcome
        obs["type"] = type

        return obs

    def get_encounter(self, idx):
        return self.generate_ebcl_data_point(idx)

    def __getitem__(self, idx):
        return self.get_encounter(idx)

    def __len__(self):
        if self.args.sampling_method == EBCLSamplingMethod.ADMISSION:
            return len(self.encounters)
        else:
            return len(self.mrn_list)


class EBCLDataLoaderCreator(BaseDataloaderCreator):
    def __init__(self, args: EBCLDataloaderConfig, shuffle=None):
        self.args = args
        logging.info("Loading Timeseries Data")
        with open(os.path.join(args.data_dir, "timeseries_dict.pkl"), "rb") as f:
            try:
                self.data_dict = pickle.load(f)
            except ModuleNotFoundError:
                self.data_dict = pd.compat.pickle_compat.load(f)
                print("Loading data via pd.compat!")
        self.shuffle = shuffle

    @classmethod
    def get_encounters(cls, args: EBCLDataloaderConfig, encounter_set: EncounterSet, split: Split):
        if args.sampling_method in ABLATION_ENCOUNTERS:
            encounters = load_ablation_encounters(args, split=split)
        elif args.sampling_method == EBCLSamplingMethod.CENSORED:
            encounters = load_censored_encounters(args, split=split)
        else:
            encounters = load_encounters(args, encounter_set=encounter_set, split=split, inpatient_only=True)
        return encounters

    def get_dataset(self, encounter_set, split):
        # Load and shuffle encounters
        frac = 1
        # Fixed on Sept. 17 - subsample validation (not test set)
        if split == Split.TRAIN or split == Split.VAL:
            assert (
                self.args.subsample <= 1 and self.args.subsample > 0
            ), "subsample should be between 0 and 1"
            frac = self.args.subsample

        encounters = self.get_encounters(
            self.args, encounter_set, split
        ).sample(frac=frac, random_state=self.args.seed)
        logging.info(f"len(encounters) for split {split.value}: {len(encounters)}")

        dataset = EBCLDataset(self.args, self.data_dict, encounters)
        return dataset

    def _get_dataloader(self, dataset, split):
        drop_last = split != Split.TEST
        shuffle = split != Split.TEST
        if self.shuffle is not None:
            shuffle = self.shuffle
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.args.num_cpus,
            pin_memory=True,
            collate_fn=partial(collate_triplets, self.args.modalities, self.args.half_dtype),
        )

    def get_dataloaders(self):
        train_dataset = self.get_dataset(self.args.encounter_set, Split.TRAIN)
        test_dataset = self.get_dataset(self.args.encounter_set, Split.TEST)
        val_dataset = self.get_dataset(self.args.encounter_set, Split.VAL)

        train_dataloader = self._get_dataloader(train_dataset, split=Split.TRAIN)
        test_dataloader = self._get_dataloader(test_dataset, split=Split.TEST)
        val_dataloader = self._get_dataloader(val_dataset, split=Split.VAL)
        return train_dataloader, val_dataloader, test_dataloader


def push_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        device_dict = dict()
        for k, v in batch.items():
            device_dict[k] = push_to(v, device)
        return device_dict
    else:
        return batch


def sanity_check_ablation_sampling_methods():
    args = EBCLDataloaderConfig(encounter_set=EncounterSet.SUFFICIENT, sampling_method=EBCLSamplingMethod.ADMISSION_PATIENT_LEVEL, post_los_cutoff=True)
    data_dict = pickle.load(open(os.path.join(args.data_dir, "timeseries_dict.pkl"), "rb"))
    for sampling_method in ABLATION_ENCOUNTERS:
        for split in Split:
            print(sampling_method, split)
            args = EBCLDataloaderConfig(encounter_set=EncounterSet.SUFFICIENT, sampling_method=sampling_method, post_los_cutoff=False)
            encounters = EBCLDataLoaderCreator.get_encounters(
                args,
                encounter_set=args.encounter_set,
                split=split,
            )
            print(encounters.shape)
            print(encounters.type.unique())
            _ = EBCLDataset(args, data_dict, encounters)
        print('\n')
    

if __name__ == "__main__":
    # python -m src.data.dataloaders.ebcl_dataloader
    import shutil
    import tempfile
    import torch

    sanity_check_ablation_sampling_methods()
