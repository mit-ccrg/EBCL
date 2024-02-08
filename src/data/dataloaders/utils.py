import logging
import os
import pickle
from abc import ABC, abstractmethod

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.configs.dataloader_configs import BaseDataloaderConfig, SupervisedMode, EBCLDataloaderConfig, EBCLSamplingMethod
from src.utils.data_utils import EncounterSet, Split

L1_KEYS = ["pre", "post"]
L2_KEYS = ["cat", "cont"]
L3_KEYS = ["date", "variable", "value"]


def load_mrns(args, split: Split, use_single_split=True):
    if use_single_split:
        if split == Split.TRAIN:
            with open(os.path.join(args.data_dir, "train_mrns.pkl"), "rb") as f:
                mrns = pickle.load(f)
        elif split == Split.TEST:
            with open(os.path.join(args.data_dir, "test_mrns.pkl"), "rb") as f:
                mrns = pickle.load(f)
        elif split == Split.VAL:
            with open(os.path.join(args.data_dir, "val_mrns.pkl"), "rb") as f:
                mrns = pickle.load(f)
        else:
            raise ValueError(f"Invalid split {split}")
        return mrns
    else:
        train_mrns, val_mrns, test_mrns, all_mrns = get_mrn_split(args)
        if split == Split.TRAIN:
            return train_mrns
        elif split == Split.TEST:
            return test_mrns
        elif split == Split.VAL:
            return val_mrns
        else:
            raise ValueError(f"Invalid split {split}")


def get_mrn_split(args):
    with open(os.path.join(args.data_dir, "mrns.pkl"), "rb") as f:
        mrns = pickle.load(f)
    train_mrns, non_train_mrns = train_test_split(
        mrns, test_size=0.2, random_state=args.data_seed
    )
    val_mrns, test_mrns = train_test_split(
        non_train_mrns, test_size=0.5, random_state=args.data_seed
    )

    return train_mrns, val_mrns, test_mrns, mrns


def load_ablation_encounters(args: EBCLDataloaderConfig, split: Split):
    assert args.post_los_cutoff is False
    mrns = load_mrns(args, split)
    # prefix = "post_los_cutoff_"

    assert args.encounter_set == EncounterSet.SUFFICIENT
    encounters = pd.read_pickle(
        os.path.join(args.data_dir, "sufficient_encounters.pkl")
    )
    encounters = encounters[
        ["hospital_mrn", "admit_date", "discharge_date", "type"]
    ].rename(columns=dict(admit_date="date"))

    encounters = encounters[encounters.hospital_mrn.isin(mrns)]
    # get outpatient and ED subset - ['INPATIENT', 'OUTPATIENT', 'UNKNOWN', 'OUTPATIENT_EMERGENCY']
    if args.sampling_method == EBCLSamplingMethod.OUTPATIENT:
        encounters = encounters[(encounters.type.str.endswith("OUTPATIENT"))]
    elif args.sampling_method == EBCLSamplingMethod.OUTPATIENT_AND_ED:
        # raise ValueError("We are not doing this experiment for now")
        encounters = encounters[(encounters.type.str.endswith("OUTPATIENT")) | (encounters.type.str.endswith("OUTPATIENT_EMERGENCY"))]
    elif args.sampling_method == EBCLSamplingMethod.OUTPATIENT_AND_ED_AND_UNKNOWN:
        encounters = encounters[(encounters.type.str.endswith("OUTPATIENT")) | (encounters.type.str.endswith("OUTPATIENT_EMERGENCY")) | (encounters.type.str.endswith("UNKNOWN"))]
    elif args.sampling_method == EBCLSamplingMethod.INPATIENT_NO_CUTOFF:
        encounters = encounters[encounters.type.str.endswith("INPATIENT")]
    elif args.sampling_method == EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED:
        encounters = encounters[(encounters.type.str.endswith("INPATIENT")) | (encounters.type.str.endswith("OUTPATIENT_EMERGENCY"))]
    elif args.sampling_method == EBCLSamplingMethod.INPATIENT_NO_CUTOFF_AND_ED_AND_UNKNOWN:
        encounters = encounters[(encounters.type.str.endswith("INPATIENT")) | (encounters.type.str.endswith("OUTPATIENT_EMERGENCY")) | (encounters.type.str.endswith("UNKNOWN"))]
    else:
        raise ValueError(f"Invalid sampling method {args.sampling_method}, use load_encounters function instead")

    # take subset of mrns in our cohort
    args.post_los_cutoff = True
    admission_cohort_encounter_mrns = load_encounters(args, encounter_set=args.encounter_set, split=split, inpatient_only=True).hospital_mrn.unique()
    encounters = encounters[encounters.hospital_mrn.isin(admission_cohort_encounter_mrns)]
    args.post_los_cutoff = False

    assert encounters.hospital_mrn.dtype == int
    assert encounters.date.dtype == "datetime64[ns]"

    return encounters


def load_censored_encounters(args: EBCLDataloaderConfig, split: Split):
    assert args.post_los_cutoff is False
    assert args.sampling_method == EBCLSamplingMethod.CENSORED

    mrns = load_mrns(args, split)
    encounters = pd.read_pickle(
        os.path.join(args.data_dir, "censored_encounters_{args.censor_cutoff}.pkl")
    )
    encounters = encounters[
        ["hospital_mrn", "admit_date", "discharge_date", "type"]
    ].rename(columns=dict(admit_date="date"))

    encounters = encounters[encounters.hospital_mrn.isin(mrns)]
    # we only use INPATIENT for Censored experiment
    encounters = encounters[encounters.type == "INPATIENT"]
    assert encounters.hospital_mrn.dtype == int
    assert encounters.date.dtype == "datetime64[ns]"

    return encounters


def load_encounters(
    args: BaseDataloaderConfig,
    encounter_set: EncounterSet,
    split: Split,
    inpatient_only: bool,
):
    # Use this to load inpatient with post_los_cutoff encounters
    mrns = load_mrns(args, split)
    global OUTCOME_STD, OUTCOME_MEAN
    OUTCOME_STD = None
    OUTCOME_MEAN = None
    if args.post_los_cutoff:
        prefix = "post_los_cutoff_"
    else:
        prefix = ""

    if encounter_set == EncounterSet.PROCESSED:
        assert (
            not args.post_los_cutoff
        ), "post_los_cutoff should be False for PROCESSED encounters."
        encounters = pd.read_pickle(
            os.path.join(args.data_dir, "processed_encounter.pkl")
        )
        encounters = encounters[
            ["hospital_mrn", "admit_date", "discharge_date", "type"]
        ].rename(columns=dict(admit_date="date"))
    elif encounter_set == EncounterSet.SUFFICIENT:
        if args.sampling_method == EBCLSamplingMethod.CENSORED:
            encounters = pd.read_pickle(
                os.path.join(args.data_dir, prefix + "censored_encounters_{args.censor_cutoff}.pkl")
            )
            if args.post_los_cutoff:
                raise NotImplementedError

        else:
            encounters = pd.read_pickle(
                os.path.join(args.data_dir, prefix + "sufficient_encounters.pkl")
            )
        encounters = encounters[
            ["hospital_mrn", "admit_date", "discharge_date", "type"]
        ].rename(columns=dict(admit_date="date"))

    elif encounter_set == EncounterSet.LOS:
        # assert (
        #     args.unimodal and not args.post
        # ), "LOS is only supported for unimodal pre models"
        encounters = pd.read_pickle(
            os.path.join(args.data_dir, prefix + "los_outcome.pkl")
        )
        # only keep inpatient visits
        encounters = encounters[encounters.type == "INPATIENT"]
        # outcome is length of stay for inpatient visit
        encounters["outcome"] = encounters.LOS
        encounters = encounters[
            ["hospital_mrn", "admit_date", "discharge_date", "type", "outcome"]
        ].rename(columns=dict(admit_date="date"))
        threshold = 7

    elif encounter_set == EncounterSet.MORTALITY:
        encounters = pd.read_pickle(
            os.path.join(args.data_dir, prefix + "mortality_outcome.pkl")
        )
        # remove patients missing the date of death
        encounters = encounters[encounters.date_of_death.notnull()]
        # outcome is death date minus current inpatient admit date
        encounters["outcome"] = encounters.date_of_death - encounters.discharge_date
        encounters = encounters[encounters["outcome"] > pd.Timedelta(days=0)]
        encounters = encounters[
            ["hospital_mrn", "admit_date", "discharge_date", "type", "outcome"]
        ].rename(columns=dict(admit_date="date"))
        threshold = 365.25

    elif encounter_set == EncounterSet.READMISSION:
        encounters = pd.read_pickle(
            os.path.join(args.data_dir, prefix + "readmission_outcome.pkl")
        )
        # outcome is next readmission date minus current inpatient discharge date
        encounters["outcome"] = encounters.readmit_date - encounters.discharge_date
        encounters = encounters[encounters["outcome"] > pd.Timedelta(days=0)]
        encounters = encounters[
            ["hospital_mrn", "admit_date", "discharge_date", "type", "outcome"]
        ].rename(columns=dict(admit_date="date"))
        threshold = 30
    else:
        raise ValueError(f"Invalid encounter set {encounter_set}")

    encounters = encounters[encounters.hospital_mrn.isin(mrns)]

    if args.post_los_cutoff:
        assert encounters["discharge_date"].notnull().all()
        encounters["los"] = encounters["discharge_date"] - encounters["date"]
    if "outcome" in encounters.columns:
        if encounter_set != EncounterSet.LOS:
            if (
                args.post_los_cutoff
            ):  # training on post-los cutoff encounters should not include encounters that are within post-los-cutoff
                # Nov 8, 2023: realized this is a bug, we were accidentally filtering to only include encounters with outcome > 1 day + los after discharge
                encounters = encounters[encounters.outcome > pd.Timedelta(days=1)]
        encounters["outcome"] = encounters["outcome"] / pd.Timedelta(days=1)
        encounters["raw_outcome"] = encounters[
            "outcome"
        ]  # raw_outcome is outcome in days
        if args.supervised_mode == SupervisedMode.BINARY:
            encounters["outcome"] = encounters["outcome"] > threshold
        else:
            assert args.supervised_mode == SupervisedMode.REGRESSION
            mean = encounters.outcome.mean()
            std = encounters.outcome.std()
            logging.info(
                f"Normalizing {encounter_set.value} {split.value}:: mean: {mean}, std: {std}"
            )
            encounters["outcome"] = (encounters["outcome"] - mean) / std
    assert inpatient_only
    if inpatient_only:
        encounters = encounters[encounters.type == "INPATIENT"]

    assert encounters.hospital_mrn.dtype == int
    assert encounters.date.dtype == "datetime64[ns]"
    return encounters


def get_dtype(half_dtype, l2, l3):
    if l3 == "date":
        return torch.float16 if half_dtype else torch.float32
    elif l3 == "variable":
        return torch.int32
    else:
        if l2 == "cat":
            return torch.int32
        else:
            return torch.float16 if half_dtype else torch.float32


def collate_value(batch, half_dtype, l1, l2, l3, device=None):
    lengths = [len(data[l1][l2][l3]) for data in batch]
    data_list = []
    for data in batch:
        data_list.extend(data[l1][l2][l3])
    data_tensors = torch.as_tensor(data_list, dtype=get_dtype(half_dtype, l2, l3), device=device)
    return data_tensors, lengths


def df_to_dict(df, event_time, is_cat, std_time):
    time = (df["date"] - event_time) / std_time
    value = df["value"]
    if is_cat:
        value = value.astype(int)
    return dict(
        date=time.to_list(),
        variable=df["variable"].to_list(),
        value=value.to_list(),
    )


def process_df_to_dict(df, event_time, std_time):
    cat_df = df[df["is_cat"] == 1]
    cont_df = df[df["is_cat"] == 0]
    sample_dict = dict(
        cat=df_to_dict(cat_df, event_time, is_cat=True, std_time=std_time),
        cont=df_to_dict(cont_df, event_time, is_cat=False, std_time=std_time),
    )
    return sample_dict


class BaseDataloaderCreator(ABC):
    @abstractmethod
    def get_dataset(self, encounter_set: EncounterSet, split: Split):
        pass

    @abstractmethod
    def _get_dataloader(self, dataset: torch.utils.data.Dataset, split: Split):
        pass

    def get_dataloaders(self):
        train_dataset = self.get_dataset(self.args.encounter_set, Split.TRAIN)
        test_dataset = self.get_dataset(self.args.encounter_set, Split.TEST)
        val_dataset = self.get_dataset(self.args.encounter_set, Split.VAL)

        train_dataloader = self._get_dataloader(train_dataset, split=Split.TRAIN)
        test_dataloader = self._get_dataloader(test_dataset, split=Split.TEST)
        val_dataloader = self._get_dataloader(val_dataset, split=Split.VAL)
        return train_dataloader, val_dataloader, test_dataloader