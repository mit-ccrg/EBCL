import logging
import os
import pickle
import time
from typing import Any, Mapping, Sequence
from functools import partial

import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
import torch

from src.configs.dataloader_configs import DuettDataloaderConfig, SupervisedView
from src.data.dataloaders.utils import BaseDataloaderCreator, load_encounters, load_mrns
from src.utils.data_utils import Split


def get_top_n_cat_and_cont_df(df, top_n):
    cat_df = (
        df[df["is_cat"].astype(bool)].reset_index(drop=True).drop(columns=["is_cat"])
    )
    cont_df = (
        df[~df["is_cat"].astype(bool)].reset_index(drop=True).drop(columns=["is_cat"])
    )
    # import pdb; pdb.set_trace()
    sorted_cat = cat_df.value.value_counts()
    sorted_cont = cont_df.variable.value_counts()
    top_n_cat_ptr, top_n_cont_ptr = 0, 0
    for i in range(top_n):
        if sorted_cat.iloc[top_n_cat_ptr] > sorted_cont.iloc[top_n_cont_ptr]:
            top_n_cat_ptr += 1
        else:
            top_n_cont_ptr += 1
    print(
        f"Take top {top_n_cat_ptr} categorical features and top {top_n_cont_ptr} continuous features"
    )
    cat_df = cat_df[cat_df.value.isin(sorted_cat.iloc[:top_n_cat_ptr].index)]
    cont_df = cont_df[cont_df.variable.isin(sorted_cont.iloc[:top_n_cont_ptr].index)]
    return cat_df, cont_df


def get_top_n_cat_and_cont_vars(df, train_mrns, top_n: int = 32) -> Sequence[int]:
    print(f"Extracting top {top_n} categorical and continuous variables")
    start = time.time()
    # filter to variables with at least one observation
    cat_df, cont_df = get_top_n_cat_and_cont_df(df, top_n)
    print(f"Time elapsed: {round(time.time() - start)} seconds")
    print("Getting One Hot Encodings for Categorical Variables")
    start = time.time()
    ohe_cat_df = pd.get_dummies(
        cat_df, columns=["value"], prefix="cat_val", sparse=False, dtype="float"
    )
    ohe_cat_df.drop(columns=["variable"], inplace=True)
    print(f"Time elapsed: {round(time.time() - start)} seconds")
    print("Melting categoricals")
    start = time.time()
    ohe_cat_df.columns = ohe_cat_df.columns.astype('category')
    cat_df = ohe_cat_df.melt(id_vars=["hospital_mrn", "date"], var_name="variable", value_name="value")
    assert cat_df.variable.dtype == 'category'    
    # cat_df['value'] = cat_df['value'].astype('float')
    cont_df["variable"] = (cont_df["variable"].astype("category").cat.rename_categories(lambda x: f"cont_val_{x}"))
    print(f"Time elapsed: {round(time.time() - start)} seconds")
    cat_df['variable'] = cat_df.variable.cat.remove_categories(["date", "hospital_mrn"])
    print("Merging OHE Categorical and Continuous Variables")
    # import pdb; pdb.set_trace()
    start = time.time()
    uc = union_categoricals([cont_df.variable, cat_df.variable])
    cat_df['variable'] = cat_df.variable.cat.add_categories(uc.categories[~uc.categories.isin(cat_df.variable.cat.categories)])
    cont_df['variable'] = cont_df.variable.cat.add_categories(uc.categories[~uc.categories.isin(cont_df.variable.cat.categories)])
    assert len(uc.categories) == top_n
    full_df = pd.concat([cont_df, cat_df], axis=0, copy=False, ignore_index=True)
    assert full_df.variable.dtype == 'category'
    # full_df["variable"] = full_df["variable"].astype("category")
    print(f"Time elapsed: {round(time.time() - start)} seconds")
    return full_df


def get_date_bins(df, admission_start, admission_end, args):
    # Create date bins
    date_bins = pd.date_range(
        start=admission_start, end=admission_end, periods=args.seq_len + 1
    )
    # Bin data
    df_binned_dates = pd.cut(
        df["date"],
        bins=date_bins,
        labels=range(args.seq_len),
        include_lowest=True,
        right=True,
    )
    return df_binned_dates, date_bins


def fast_aggregate(df):
    # Assuming 'variable' and 'date_bin' are already optimized types (e.g., category)

    # Group by and aggregate
    grouped = df.groupby(["variable", "date_bin"], observed=False)
    mean = grouped.mean()
    count = grouped.count()

    # You might reshape these as needed, but avoid concatenating if possible
    return mean, count


def convert_admission_df_to_variable_vs_time_matrix(
    df: pd.DataFrame,
    args: DuettDataloaderConfig,
    admission_start: pd.Timestamp,
    admission_end: pd.Timestamp,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Generates a matrix of variables vs time bins. Each row is a variable and each column is a time bin.

    The number of time bins are defined by the seq_len argument in the DuettDataloaderConfig.

    Args:
        df (pd.DataFrame): Dataframe of all observations over a specific inpatient admission
        args (DuettDataloaderConfig): defines the seq_len
        admission_start (pd.Timestamp): start date of admission (or first observation)
        admission_end (pd.Timestamp): end date of admission (discharge time)
        top_n_vars (Sequence[int]): top n variables in the dataset to filter observations to
        rand (np.random.Generator): random number generator, used for random aggregation of
            multiple observations for same event over same time bin

    Returns:
        pd.DataFrame: matrix of variables vs time bins, used as input to Duett model
    """
    # Create date bins
    df["date_bin"], date_bins = get_date_bins(df, admission_start, admission_end, args)

    mean, count = fast_aggregate(df[["variable", "date_bin", "value"]])
    cont_pivot_df = mean.unstack(level="date_bin")
    freq_pivot_df = count.unstack(level="date_bin")

    cont_pivot_df.columns = cont_pivot_df.columns.get_level_values(1)
    freq_pivot_df.columns = freq_pivot_df.columns.get_level_values(1)

    # Fill in missing variables and missing time bins with NaNs -- takes .4/1.8 of the time
    cont_df = fill_out_nan(cont_pivot_df, columns, args.seq_len)
    cont_count_df = fill_out_nan(freq_pivot_df, columns, args.seq_len)
    # nans can occur in value count when there are 0 elements, so fill with 0's
    cont_count_df.fillna(0, inplace=True)
    return cont_df, cont_count_df, date_bins


def fill_out_nan(df, columns, seq_len):
    missing_date_bins = [i for i in range(seq_len) if i not in df.columns]
    missing_variables = [var for var in columns if var not in df.index]
    df.loc[:, missing_date_bins] = np.NAN
    df = df[df.columns.sort_values(ascending=True)]
    nan_df = pd.DataFrame(np.NAN, index=missing_variables, columns=df.columns)
    df = pd.concat([df, nan_df]).sort_index()
    return df


class DuettDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args: DuettDataloaderConfig,
        data_dict: Mapping[int, pd.DataFrame],
        encounters: pd.DataFrame,
        columns: Sequence[str],
    ):
        """Initialize Admissions dataset

        Args:
            args (DuettDataloaderConfig): configuration for the dataloader
            data_dict (Mapping[int, pd.DataFrame]): mapping from patient MRN to dataframe of observations
            encounters (pd.DataFrame): dataframe of admission encounters
            columns:
        """
        self.args = args
        assert isinstance(args, DuettDataloaderConfig), type(args)
        self.data_dict = data_dict
        self.encounters = encounters
        self.columns = columns
        # self.top_n_vars, self.cat_max = get_most_common_cont_variables(
        #     args.data_dir, args.num_events
        # )
        # with open(args.data_dir + "/mapping.pkl", "rb") as f:
        #     mappings = pickle.load(f)
        #     self.cat_vars = sorted(list(mappings["cat_var_map"].keys()))

        self.std_time = self.encounters.date.std()
        self.deidentified_keys = [
            "data",
            "freq",
            "times",
            "outcome",
            "raw_outcome",
        ]

    def get_admission_data(self, idx):
        # TODO restrict to 512 most recent encounters for Post
        row = self.encounters.iloc[idx]

        subset_df = self.data_dict[row["hospital_mrn"]]
        # only get the top n variables (duett was made to take a reduced number of variables)
        subset_df = subset_df[subset_df.variable.isin(self.columns)]

        admission_start = row["date"]
        admission_end = row["date"] + row["los"] + pd.Timedelta(days=1)
        # filter to data during admission
        post_filtered_indexes = subset_df["date"] > admission_start
        post_filtered_indexes &= subset_df["date"] <= admission_end
        post_filtered_rows = subset_df[post_filtered_indexes].copy()
        return post_filtered_rows, row, admission_start, admission_end

    def get_pre_discharge_data(self, idx):
        # TODO restrict to 512 most recent encounters for Pre and Post
        row = self.encounters.iloc[idx]

        subset_df = self.data_dict[row["hospital_mrn"]]
        # only get the top n variables (duett was made to take a reduced number of variables)
        subset_df = subset_df[subset_df.variable.isin(self.columns)]

        start_time = min(subset_df["date"].min(), row["date"])
        if (
            start_time != start_time
        ):  # if no data is available, set start time to admission time
            start_time = row["date"]
        admission_end = row["date"] + row["los"] + pd.Timedelta(days=1)
        # filter to data during admission
        post_filtered_indexes = subset_df["date"] <= admission_end
        post_filtered_rows = subset_df[post_filtered_indexes].copy()
        return post_filtered_rows, row, start_time, admission_end

    def get_pre_admission_data(self, idx):
        # TODO restrict to 512 most recent encounters for Pre and Post
        row = self.encounters.iloc[idx]

        subset_df = self.data_dict[row["hospital_mrn"]]
        # only get the top n variables (duett was made to take a reduced number of variables)
        subset_df = subset_df[subset_df.variable.isin(self.columns)]

        start_time = min(subset_df["date"].min(), row["date"])
        if (
            start_time != start_time
        ):  # if no data is available, set start time to admission time
            start_time = row["date"] - pd.Timedelta(days=1)
        end_time = row["date"]
        if not start_time < end_time:
            start_time = end_time - pd.Timedelta(days=1)
        # filter to data during admission
        pre_filtered_indexes = subset_df["date"] <= end_time
        pre_filtered_rows = subset_df[pre_filtered_indexes].copy()
        return pre_filtered_rows, row, start_time, end_time

    def generate_duett_data_point(self, idx: int) -> Mapping[str, Any]:
        """Given admission index, gets duett model input data.

        Given admission index, retrieves the admission observations, converts the observations
        to a variable vs time bin matrix, and returns the matrix as the duett model input data.

        Args:
            idx: index of the encounter to retrieve
        Returns:
            Mapping[str, Any]: Model ready input data.
        """
        if self.args.modalities == SupervisedView.POST:
            admission_df, row, start_time, end_time = self.get_admission_data(idx)
        elif self.args.modalities == SupervisedView.PRE_AND_POST:
            admission_df, row, start_time, end_time = self.get_pre_discharge_data(idx)
        elif self.args.modalities == SupervisedView.PRE:
            admission_df, row, start_time, end_time = self.get_pre_admission_data(idx)
        else:
            raise ValueError(f"Invalid modality: {self.args.modalities}")

        encounter_date = row["date"]

        assert start_time < end_time, f"start_time: {start_time}, end_time: {end_time}"

        (
            data_df,
            count_df,
            date_bins,
        ) = convert_admission_df_to_variable_vs_time_matrix(
            admission_df,
            self.args,
            start_time,
            end_time,
            self.columns,
        )
        data_ar, count_ar = (
            data_df.to_numpy(),
            count_df.to_numpy(),
        )
        assert data_ar.shape == count_ar.shape
        assert data_ar.shape[1] == self.args.seq_len
        assert data_ar.shape[0] == len(self.columns)
        obs = dict(
            hospital_mrn=row["hospital_mrn"],
            encounter_date=encounter_date,
            data=data_ar,
            freq=count_ar,
            times=(
                ((date_bins - date_bins[0]) / self.std_time)[1:]
            ).to_numpy(),  # get the relative end times for each time bin
        )

        outcome = row.get("outcome")
        obs["outcome"] = outcome
        obs["raw_outcome"] = row.get("raw_outcome")
        obs["type"] = row["type"]
        return obs

    def get_encounter(self, idx):
        return self.generate_duett_data_point(idx)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self.get_encounter(idx)
        else:
            raise NotImplementedError("Only int indexing is supported")
            return [self.get_encounter(i) for i in idx]

    def __len__(self):
        return self.encounters.shape[0]


def collate_duett_triplets(half_dtype, batch, device=None):
    collated_batch = dict()
    try:
        mrns = [each["hospital_mrn"] for each in batch]
        collated_batch["hospital_mrn"] = mrns

        encounter_date = [each["encounter_date"] for each in batch]
        collated_batch["encounter_date"] = encounter_date
    except:
        pass

    if half_dtype:
        dtype = torch.float16
    else:
        dtype = torch.float32
    for key in ["data", "freq", "times", "outcome"]:
        if key == "outcome" and batch[0][key] is None:
            continue
        collated_batch[key] = torch.utils.data.default_collate(
            [each[key] for each in batch]
        ).type(dtype)
    return collated_batch


def process_data_dict(data_dict, args):
    df = pd.concat(data_dict.values())
    train_mrns = load_mrns(args, Split.TRAIN)
    full_df = get_top_n_cat_and_cont_vars(df, train_mrns, top_n=args.num_events)
    processed_data_dict = dict(list(full_df.groupby("hospital_mrn")))
    columns = sorted(full_df.variable.unique().astype(str))
    return processed_data_dict, columns


class DuettDataLoaderCreator(BaseDataloaderCreator):
    def __init__(self, args: DuettDataloaderConfig, shuffle):
        self.args = args
        logging.info("Loading Timeseries Data")

        data_dir = os.path.join(args.data_dir, f"duett_timeseries_dict_{args.num_events}.pkl")
        if os.path.exists(data_dir):
            logging.warn("Loading Data at {}".format(data_dir))
            with open(data_dir, "rb") as f:
                try:
                    data_dict = pickle.load(f)
                except ModuleNotFoundError:
                    data_dict = pd.compat.pickle_compat.load(f)
                    print("Loading data via pd.compat!")
            self.data_dict, self.columns = data_dict['data'], data_dict['columns']
        else:
            logging.warn("Storing Data at {}".format(data_dir))
            with open(os.path.join(args.data_dir, "timeseries_dict.pkl"), "rb") as f:
                try:
                    data_dict = pickle.load(f)
                except ModuleNotFoundError:
                    data_dict = pd.compat.pickle_compat.load(f)
                    print("Loading data via pd.compat!")
            self.data_dict, self.columns = process_data_dict(data_dict, args)
            data_dict = dict(data=self.data_dict, columns=self.columns)
            if args.store_data:
                with open(data_dir, "wb") as f:
                    pickle.dump(data_dict, f)

    def get_dataset(self, encounter_set, split):
        frac = 1
        encounters = load_encounters(
            self.args, encounter_set, split, self.args.inpatient_only
        ).sample(frac=frac, random_state=self.args.seed)
        failed_rows = []
        for i in range(encounters.shape[0]):
            row = encounters.iloc[i]
            mrn = row["hospital_mrn"]
            if mrn not in self.data_dict:
                print(f"MRN: {mrn} not found in data_dict")
                failed_rows.append(i)
        assert len(failed_rows) <= 3, f"Failed rows: {failed_rows}"
        encounters = (
            encounters.reset_index(drop=True).drop(failed_rows).reset_index(drop=True)
        )
        logging.info(f"len(encounters) for split {split.value}: {len(encounters)}")
        dataset = DuettDataset(self.args, self.data_dict, encounters, self.columns)

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
            collate_fn=partial(collate_duett_triplets, self.args.half_dtype),
        )


if __name__ == "__main__":
    # python -m src.data.dataloaders.duett_dataloader
    import shutil
    import tempfile
    import torch

    from src.utils.data_utils import EncounterSet
    
    from tests.test_dataloaders import generate_synthetic_data


    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory for tests: {temp_dir}")

    generate_synthetic_data(temp_dir)

    for modality in SupervisedView:
        if modality == SupervisedView.EARLY_FUSION:
            continue
        args = DuettDataloaderConfig(encounter_set=EncounterSet.MORTALITY, data_dir=temp_dir, num_events=8, modalities=modality)
        dc = DuettDataLoaderCreator(args)
        dataset = dc.get_dataset(EncounterSet.MORTALITY, Split.TRAIN)
        output = dataset[0]
        assert isinstance(output, dict)
    # Cleanup after all tests have run
    shutil.rmtree(temp_dir)
    print(f"Deleted temporary directory for tests: {temp_dir}")