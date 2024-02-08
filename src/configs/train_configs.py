import dataclasses
import enum
import logging
import os
import re
import shutil
from dataclasses import field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Type, TypeVar

from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import MISSING, OmegaConf

from src.configs.dataloader_configs import (
    BaseDataloaderConfig,
    DuettDataloaderConfig,
    StratsDataloaderConfig,
    EBCLDataloaderConfig,
    OCPDataloaderConfig,
    SupervisedView,
)
from src.configs.model_configs import DuettModelConfig, StratsModelConfig
from src.configs.utils import hydra_dataclass
from src.utils.data_utils import EncounterSet
from src.utils.process_utils import file_path


@dataclasses.dataclass
class LoggerConfig:
    """Logger Configuration for a `Dataset`.

    Attributes:
        val_iter: run and log validation set performance every val_iter training epochs
        max_val_iter: maximum number of validation set batches to run over
        log_iter: log training loss every log_iter training batches
        train_iter_n: Number of batches to average training metrics over (performed once each val-iter) before logging
        save_iter: save model every save_iter training epochs
        save_dir: directory to save results to

    """

    name: str = "test"
    reset: bool = False
    val_iter: int = 1
    max_val_iter: Optional[int] = None
    train_iter_n: int = 1
    save_iter: int = 10
    save_dir: str = os.path.join(Path.home(), "results")
    batch_size: int = 256


def hydra_train_dataclass(dataclass: Any) -> Any:
    """Decorator that allows you to use a dataclass as a hydra config via the `ConfigStore`

    Adds the decorated dataclass as a `Hydra StructuredConfig object`_ to the `Hydra ConfigStore`_.
    The name of the stored config in the ConfigStore is the snake case version of the CamelCase class name.

    .. _Hydra StructuredConfig object: https://hydra.cc/docs/tutorials/structured_config/intro/

    .. _Hydra ConfigStore: https://hydra.cc/docs/tutorials/structured_config/config_store/
    """

    dataclass = dataclasses.dataclass(dataclass)

    name = dataclass.__name__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    cs = ConfigStore.instance()
    cs.store(group="dataloader_config", name="ebcl", node=EBCLDataloaderConfig)
    cs.store(group="dataloader_config", name="ocp", node=OCPDataloaderConfig)
    cs.store(group="dataloader_config", name="duett", node=DuettDataloaderConfig)
    cs.store(group="dataloader_config", name="strats", node=StratsDataloaderConfig)
    cs.store(name=name, node=dataclass)

    cs.store(group="model_config", name="ebcl", node=StratsModelConfig)
    cs.store(group="model_config", name="ocp", node=StratsModelConfig)
    cs.store(group="model_config", name="strats", node=StratsModelConfig)
    cs.store(group="model_config", name="duett", node=DuettModelConfig)
    cs.store(name=name, node=dataclass)

    return dataclass


defaults = [
    # Load the config "mysql" from the config group "db"
    "_self_",
    {"dataloader_config": "ebcl"},
    {"model_config": "ebcl"},
]


@dataclasses.dataclass
class BaseTrainConfig:
    name: str = MISSING
    # this is unfortunately verbose due to @dataclass limitations
    defaults: List[Any] = field(default_factory=lambda: defaults)

    # Hydra will populate this field based on the defaults list
    dataloader_config: BaseDataloaderConfig = MISSING
    model_config: Any = MISSING
    logger_config: LoggerConfig = LoggerConfig(name="default")
    early_stop_tol: int = 3
    max_val_iter: Optional[int] = None
    batch_size: int = 256

    def __post_init__(self):
        # check that output directory is empty
        self.logger_config.name = self.name
        self.logger_config.batch_size = self.batch_size
        if not isinstance(self.dataloader_config, str):
            self.dataloader_config.batch_size = self.batch_size
        if not isinstance(self.model_config, str):
            self.model_config.batch_size = self.batch_size
        if self.logger_config.reset:
            shutil.rmtree(self.get_save_dir(), ignore_errors=True)
        if os.path.exists(self.logger_config.save_dir):
            logging.warn(f"{self.logger_config.save_dir} already exists")
        else:
            os.makedirs(self.logger_config.save_dir)
        if isinstance(self.dataloader_config, DuettDataloaderConfig):
            assert isinstance(self.model_config, DuettModelConfig)
            assert self.dataloader_config.num_events == self.model_config.num_event
            assert self.dataloader_config.seq_len == self.model_config.seq_len
        # assert not os.path.exists(self.get_save_dir())

    def get_save_dir(self):
        return os.path.join(self.logger_config.save_dir, self.name)


class PretrainMethod(enum.Enum):
    EBCL = "ebcl"
    RAND = "rand"
    OCP = "ocp"
    DUETT = "duett"


class SupervisedMethod(enum.Enum):
    EBCL = "ebcl"
    OCP = "ocp"
    DUETT = "duett"
    RAND = "rand"
    FROZEN = "frozen"


@hydra_dataclass
class PretrainCheckpoints:
    ckpt_1: Optional[str]
    ckpt_2: Optional[str]
    ckpt_3: Optional[str]
    idx: int = 0

    def __getitem__(self, idx):
        assert idx in [0, 1, 2]
        if idx == 0:
            return self.ckpt_1
        elif idx == 1:
            return self.ckpt_2
        else:
            return self.ckpt_3


@hydra_train_dataclass
class PretrainConfig(BaseTrainConfig):
    """Training Configuration for a `Dataset`.

    Attributes:
        name: name of experiment. results are stored in `logger.save_dir/name`
        debug: debug mode - run one batch and 4 epochs
        gpus: gpus to use for training
        reset: how to reset the environment
        seed: global seed to set before training
        inpatient_only: only use inpatient encounters
        early_fusion: use early fusion - note that this halves the sequence length
        epochs: max epochs to train for.
        batch_size: batch size.
        lr: learning rate for Adam optimizer
        early_stop_tol: how many epochs to wait for no validation set improvement before early stopping.
        post: Use Post vs. Post for unimodal contrastive learning, Pre data is default.
        train_method: TrainMethod
        logger (LoggerConfig): configuration for the logger
    """

    debug: bool = False
    gpus: Optional[List[int]] = None
    reset: bool = False
    seed: int = 1
    epochs: int = 100
    lr: float = 1e-3
    early_stop_tol: int = 3
    pretrain_method: PretrainMethod = PretrainMethod.EBCL
    logger: LoggerConfig = LoggerConfig()


@hydra_train_dataclass
class SupervisedTrainConfig(PretrainConfig):
    """SupervisedTrainConfig Configuration for a `Dataset`.

    Attributes:
        supervised_method: method to use for supervised training (EBCL, OCP, Frozen, Supervised)
        unimodal: unimodal
        pretrain_ckpts: pretrained ckpt to load
    """

    modalities: Optional[SupervisedView] = None
    supervised_method: SupervisedMethod = SupervisedMethod.EBCL
    pretrain_ckpt: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if (
            self.supervised_method != SupervisedMethod.DUETT
            and self.dataloader_config is not MISSING
        ):
            self.dataloader_config.modalities = self.modalities
            self.model_config.modalities = self.modalities


EBCL_PRETRAIN_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.EBCL",
    "dataloader_config=ebcl",
    "model_config=ebcl",
    "model_config.architecture_size=Architecture.MEDIUM",
    "dataloader_config.sampling_method=EBCLSamplingMethod.ADMISSION",
    "dataloader_config.modalities=SupervisedView.PRE_AND_POST",
    "model_config.modalities=SupervisedView.PRE_AND_POST",
]
EBCL_SUPERVISED_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.EBCL",
    "supervised_method=SupervisedMethod.EBCL",
    "dataloader_config=ebcl",
    "model_config=ebcl",
    "model_config.architecture_size=Architecture.MEDIUM",
    "dataloader_config.sampling_method=EBCLSamplingMethod.ADMISSION",
    "modalities=SupervisedView.PRE_AND_POST",
    "dataloader_config.modalities=SupervisedView.PRE_AND_POST",
    "model_config.modalities=SupervisedView.PRE_AND_POST",
]

OCP_PRETRAIN_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.OCP",
    "dataloader_config=ocp",
    "model_config=ocp",
    "model_config.architecture_size=Architecture.MEDIUM",
    "early_stop_tol=10",
]

OCP_SUPERVISED_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.OCP",
    "supervised_method=SupervisedMethod.OCP",
    "dataloader_config=ebcl",  # we use EBCL input data for OCP finetuning
    "model_config=ocp",
    "model_config.architecture_size=Architecture.MEDIUM",
    "dataloader_config.sampling_method=EBCLSamplingMethod.ADMISSION",
    "modalities=SupervisedView.PRE_AND_POST",
    "dataloader_config.modalities=SupervisedView.PRE_AND_POST",
    "model_config.modalities=SupervisedView.PRE_AND_POST",
]

RAND_SUPERVISED_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.RAND",
    "supervised_method=SupervisedMethod.RAND",
    "dataloader_config=ebcl",  # we use EBCL input data for RAND supervised training
    "model_config=ebcl",
    "model_config.architecture_size=Architecture.MEDIUM",
    "dataloader_config.sampling_method=EBCLSamplingMethod.ADMISSION",
    "modalities=SupervisedView.PRE_AND_POST",
    "dataloader_config.modalities=SupervisedView.PRE_AND_POST",
    "model_config.modalities=SupervisedView.PRE_AND_POST",
]

DUETT_PRETRAIN_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.DUETT",
    "dataloader_config=duett",
    "model_config=duett",
]


DUETT_SUPERVISED_OVERRIDES: List[str] = [
    "pretrain_method=PretrainMethod.DUETT",
    "supervised_method=SupervisedMethod.DUETT",
    "dataloader_config=duett",
    "model_config=duett",
]


class Experiment(enum.Enum):
    EBCL_PRETRAIN = "ebcl_pretrain"
    EBCL_SUPERVISED = "ebcl_supervised"
    OCP_PRETRAIN = "ocp_pretrain"
    OCP_SUPERVISED = "ocp_supervised"
    RAND_SUPERVISED = "rand_supervised"
    DUETT_PRETRAIN = "duett_pretrain"
    DUETT_SUPERVISED = "duett_supervised"


PRETRAIN_OVERRIDES: Mapping[Experiment, List[str]] = {
    Experiment.EBCL_PRETRAIN: EBCL_PRETRAIN_OVERRIDES,
    Experiment.OCP_PRETRAIN: OCP_PRETRAIN_OVERRIDES,
    Experiment.DUETT_PRETRAIN: DUETT_PRETRAIN_OVERRIDES,
}


SUPERVISED_OVERRIDES: Mapping[Experiment, List[str]] = {
    Experiment.EBCL_SUPERVISED: EBCL_SUPERVISED_OVERRIDES,
    Experiment.OCP_SUPERVISED: OCP_SUPERVISED_OVERRIDES,
    Experiment.RAND_SUPERVISED: RAND_SUPERVISED_OVERRIDES,
    Experiment.DUETT_SUPERVISED: DUETT_SUPERVISED_OVERRIDES,
}


def get_pretrain_exp_config(
    name: str,
    exp: Experiment,
    encounter_set: EncounterSet,
    other_overrides: List[str] = [],
) -> PretrainConfig:
    with initialize(version_base=None):
        # config is relative to a module
        cfg = compose(
            config_name="pretrain_config",
            overrides=PRETRAIN_OVERRIDES[exp]
            + [f"name={name}", f"dataloader_config.encounter_set={str(encounter_set)}"]
            + other_overrides,
        )
        # instantiate the hydra config with _convert_="object", otherwise dataclasses are converted to dicitonaries
        print(cfg["logger_config"])
        cfg = instantiate(cfg, _convert_="object")
        return cfg


def get_supervised_exp_config(
    name: str,
    exp: Experiment,
    encounter_set: EncounterSet,
    other_overrides: List[str] = [],
    pretrain_ckpt: Optional[str] = None,
) -> SupervisedTrainConfig:
    overrides = SUPERVISED_OVERRIDES[exp] + [
        f"name={name}",
        f"dataloader_config.encounter_set={str(encounter_set)}",
    ]
    if pretrain_ckpt is not None:
        overrides += [f"pretrain_ckpt={pretrain_ckpt}"]
    with initialize(version_base=None):
        # config is relative to a module
        cfg = compose(
            config_name="supervised_train_config", overrides=overrides + other_overrides
        )
        # instantiate the hydra config with _convert_="object", otherwise dataclasses are converted to dicitonaries
        cfg = instantiate(cfg, _convert_="object")
        return cfg


def save_config(cfg: BaseTrainConfig, path: str) -> None:
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)


# T = TypeVar("T", bound=BaseTrainConfig)


def load_config(path: str, exp: Experiment) -> BaseTrainConfig:
    with open(path, "r"):
        cfg = OmegaConf.load(path)
    if "supervised_method" in cfg.keys():
        schema = get_supervised_exp_config(
            "name",
            exp,
            EncounterSet.READMISSION,
            [],
            None,
        )
        # config_class = SupervisedTrainConfig
    else:
        schema = get_pretrain_exp_config("name", exp, EncounterSet.READMISSION)
        # config_class = PretrainConfig
    with open(path, "r"):
        cfg = OmegaConf.load(path)
    # schema = OmegaConf.create(config_class)
    merge = OmegaConf.merge(schema, cfg)
    loaded_cfg = OmegaConf.to_object(merge)
    return loaded_cfg
