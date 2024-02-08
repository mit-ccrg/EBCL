import logging
import os
import pickle

import pandas as pd

MORTALITY_COL = "Mortality"
READMISSION_COL = "Readmission"
ADMISSION_COL = "Admission"
COL_RENAME = dict(
    x-x-x_x-x-x_hash="x-x-x_x-x-x",
    empi_hash="empi",
)


def load_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Unrecognized file extension for {path}")
    df.rename(columns=COL_RENAME, inplace=True)
    return df


def dir_path(string):
    """Checks that string is a valid directory path

    Args:
        string (str): input string

    Raises:
        NotADirectoryError: string is not a valid directory path

    Returns:
        str: string if it is valid directory path
    """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def file_path(string):
    """Checks that string is a valid file path

    Args:
        string (str): input string

    Raises:
        FileNotFoundError: string is not a valid file path

    Returns:
        str: string if it is valid file path
    """
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)


def file_path_list(string):
    """Checks that string is a valid list of file paths

    Args:
        string (str): input string

    Raises:
        FileNotFoundError: string is not a valid file path

    Returns:
        str: string if it is valid file path
    """
    files = string.split(",")
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError(file)
    return files


def store_args(args, output_dir):
    config = vars(args)
    config_path = os.path.join(output_dir, "config.pkl")
    if os.path.exists(config_path):
        logging.warn(f"config file already exists at {config_path}")
    with open(config_path, "wb") as outfile:
        pickle.dump(config, outfile)
