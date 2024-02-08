import dataclasses
import enum
from typing import Optional

from omegaconf import MISSING

from src.configs.utils import hydra_dataclass
from src.utils.data_utils import EncounterSet


class SupervisedView(enum.Enum):
    PRE = "pre"
    POST = "post"
    EARLY_FUSION = "early_fusion"
    PRE_AND_POST = "pre_and_post"


class EBCLSamplingMethod(enum.Enum):
    ADMISSION = "admission"
    ADMISSION_PATIENT_LEVEL = "admission_patient_level"
    RANDOM = "random"  # implicitly is patient level
    ADMISSION_RANDOM_SAMPLE = "admission_random_sample"  # implicitly is patient level
    CENSORED = "censored" # implicitly is patient level, censor IQR1 of Pre/Post observations from the event
    RANDOM_WINDOW = "random_window"  # implicitly is patient level
    OUTPATIENT = "outpatient"  # implicitly is patient level
    OUTPATIENT_AND_ED = "outpatient_and_ed"  # implicitly is patient level
    OUTPATIENT_AND_ED_AND_UNKNOWN = "outpatient_and_ed_and_unknown"  # implicitly is patient level
    INPATIENT_NO_CUTOFF = "inpatient_no_cutoff"  # implicitly is patient level
    INPATIENT_NO_CUTOFF_AND_ED = "inpatient_no_cutoff_and_ed"  # implicitly is patient level
    INPATIENT_NO_CUTOFF_AND_ED_AND_UNKNOWN = "inpatient_no_cutoff_and_ed_and_unknown"  # implicitly is patient level


class SupervisedMode(enum.Enum):
    BINARY = "binary"
    REGRESSION = "regression"


class CacheMode(enum.Enum):
    LOAD_CACHE = "load_cache"
    CREATE_CACHE = "cache"
    NO_CACHE = "no_cache"


@dataclasses.dataclass
class BaseDataloaderConfig:
    pretrain: bool = True
    inpatient_only: bool = True
    encounter_set: EncounterSet = MISSING
    supervised_mode: SupervisedMode = SupervisedMode.BINARY
    seed: int = 926
    data_dir: str = "x-x-xhf_cohort/final/"
    post_cutoff: Optional[int] = None
    post_los_cutoff: bool = True
    pre_cutoff: Optional[int] = None
    subsample: float = 1
    batch_size: int = 256
    num_cpus: int = 8
    cache_dir: str = "x-x-xhf_cohort/final/cache/"
    cache_mode: CacheMode = CacheMode.NO_CACHE
    half_dtype: bool = False
    modalities: SupervisedView = SupervisedView.PRE_AND_POST


@hydra_dataclass
class DuettDataloaderConfig(BaseDataloaderConfig):
    """Dataloader Configuration

    Attributes:
        TODO(x-x-x)
    """
    store_data: bool = False
    seq_len: int = 32
    num_events: int = 32


@hydra_dataclass
class OCPDataloaderConfig(BaseDataloaderConfig):
    """Dataloader Configuration

    Attributes:
        seed: random seed for shuffling encounters
        data_dir: directory to save data to
        max_obs: maximum number of observations per encounter
        min_obs: minimum number of observations per encounter
        num_cpus: number of cpus to use for multiprocessing
        subsample: subsampling percentage for fewshot finetuning
    """

    inpatient_only: bool = True
    max_obs: int = 512
    min_obs: int = 16


@hydra_dataclass
class EBCLDataloaderConfig(BaseDataloaderConfig):
    """Dataloader Configuration

    Attributes:
        sampling_method: sampling method to generate data using
        seed: random seed for shuffling encounters
        data_dir: directory to save data to
        max_obs: maximum number of observations per encounter
        min_obs: minimum number of observations per encounter
        num_cpus: number of cpus to use for multiprocessing
        post_cutoff: minimum number of days between encounters
        post_los_cutoff: Cutoff post data at LOS + 1 day.
        pre_cutoff: minimum number of days between encounters
        encounter_set: Encounters to generate data for.
        subsample: subsampling percentage for fewshot finetuning
        supervised_mode: Train for Binary or regression supervised task.
    """

    sampling_method: EBCLSamplingMethod = EBCLSamplingMethod.ADMISSION
    max_obs: int = 512
    min_obs: int = 16
    ocp_and_strats_ablation: bool = False
    pre_obs_censor: int = 260
    post_obs_censor: int = 60
    censor_cutoff: float = 25

@hydra_dataclass
class StratsDataloaderConfig(BaseDataloaderConfig):
    """Dataloader Configuration

    Attributes:
        max_obs: sampling method to generate data using
        min_obs: random seed for shuffling encounters
        preview_time: time window for starting forecast
        max_obs: maximum number of observations per encounter
        min_obs: minimum number of observations per encounter
        num_cpus: number of cpus to use for multiprocessing
        post_cutoff: minimum number of days between encounters
        post_los_cutoff: Cutoff post data at LOS + 1 day.
        pre_cutoff: minimum number of days between encounters
        encounter_set: Encounters to generate data for.
        subsample: subsampling percentage for fewshot finetuning
        supervised_mode: Train for Binary or regression supervised task.
    """

    max_obs: int = 512
    min_obs: int = 16
    forecast_window: int = 30
    n_cat_variable: int = 52 # variables 52 and up are continuous
    n_cat_value: int = 7761
    n_variable: int = 3275
    check: bool = False


@hydra_dataclass
class GptDataloaderConfig(BaseDataloaderConfig):
    """Dataloader Configuration

    Attributes:
        max_obs: sampling method to generate data using
        min_obs: random seed for shuffling encounters
        preview_time: time window for starting forecast
        max_obs: maximum number of observations per encounter
        min_obs: minimum number of observations per encounter
        num_cpus: number of cpus to use for multiprocessing
        post_cutoff: minimum number of days between encounters
        post_los_cutoff: Cutoff post data at LOS + 1 day.
        pre_cutoff: minimum number of days between encounters
        encounter_set: Encounters to generate data for.
        subsample: subsampling percentage for fewshot finetuning
        supervised_mode: Train for Binary or regression supervised task.
    """
    max_obs: int = 512
    min_obs: int = 16