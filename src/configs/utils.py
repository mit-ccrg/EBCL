import dataclasses
import json
import re
from pathlib import Path
from typing import Any, TypeVar

import hydra


def hydra_dataclass(dataclass: Any) -> Any:
    """Decorator that allows you to use a dataclass as a hydra config via the `ConfigStore`

    Adds the decorated dataclass as a `Hydra StructuredConfig object`_ to the `Hydra ConfigStore`_.
    The name of the stored config in the ConfigStore is the snake case version of the CamelCase class name.

    .. _Hydra StructuredConfig object: https://hydra.cc/docs/tutorials/structured_config/intro/

    .. _Hydra ConfigStore: https://hydra.cc/docs/tutorials/structured_config/config_store/
    """

    dataclass = dataclasses.dataclass(dataclass)

    name = dataclass.__name__
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()

    cs = hydra.core.config_store.ConfigStore.instance()
    cs.store(name=name, node=dataclass)

    return dataclass
