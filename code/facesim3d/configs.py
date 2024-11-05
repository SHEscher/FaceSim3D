"""
Configuration for FaceSim3D project.

**Note**: store private configs in the same folder as `config.toml`, namely: `./[PRIVATE_PREFIX]_configs.toml`

Author: Simon M. Hofmann | <[firstname].[lastname]@cbs.mpg.de> | 2022
"""

# %% Imports
from __future__ import annotations

import logging.config
import os
from pathlib import Path
from typing import Any

import toml

# %% Config class & functions ><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def _iter_nested_dicts(nested_dict: dict[str, Any]) -> Any:
    """Create generator iterating over values in nested dicts."""
    for value in nested_dict.values():
        if isinstance(value, dict):
            yield from _iter_nested_dicts(value)
        else:
            yield value


def _create_parent_dirs(config_as_dict: dict[str, Any]) -> None:
    """Create parent dirs of log files."""
    for value in _iter_nested_dicts(config_as_dict):
        if isinstance(value, str) and value.endswith(".log"):
            Path(PROJECT_ROOT, value).parent.mkdir(parents=True, exist_ok=True)


class CONFIG:
    """Configuration object."""

    def __init__(self, config_dict: dict | None = None) -> None:
        """Initialise CONFIG class object."""
        if config_dict is not None:
            self.update(config_dict)

    def __repr__(self) -> str:
        """Implement __repr__ of CONFIG."""
        str_out = "CONFIG("
        list_attr = [k for k in self.__dict__ if not k.startswith("_")]
        ctn = 0  # counter for visible attributes only
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                # ignore hidden attributes
                continue
            ctn += 1
            str_out += f"{key}="
            if isinstance(val, CONFIG):
                str_out += str(val)
            else:
                str_out += f"'{val}'" if isinstance(val, str) else f"{val}"

            str_out += ", " if ctn < len(list_attr) else ""
        return str_out + ")"

    def update(self, new_configs: dict[str, Any]) -> None:
        """Update the config object with new entries."""
        for k, val in new_configs.items():
            if isinstance(val, list | tuple):
                setattr(self, k, [CONFIG(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, k, CONFIG(val) if isinstance(val, dict) else val)

    def show(self, indent: int = 0):
        """
        Display the nested configuration information.

        :param indent: The number of tabs to use for indentation (default: 0)
        :type indent: int
        :return: None
        """
        for key, val in self.__dict__.items():
            if isinstance(val, CONFIG):
                print("\t" * indent + f"{key}:")
                val.show(indent=indent + 1)
            else:
                _val = val.replace("\n", "\\n").replace("\t", "\\t") if isinstance(val, str) else val
                print("\t" * indent + f"{key}: " + (f"'{_val}'" if isinstance(val, str) else f"{val}"))

    def asdict(self):
        """Convert the config object to dict."""
        dict_out = {}
        for key, val in self.__dict__.items():
            if isinstance(val, CONFIG):
                dict_out.update({key: val.asdict()})
            else:
                dict_out.update({key: val})
        return dict_out

    def update_paths(self, parent_path: str | None = None, for_logging: bool = False):
        """Update relative paths to PROJECT_ROOT dir."""
        # Use project root dir as the parent path if it is not specified
        parent_path = self.PROJECT_ROOT if hasattr(self, "PROJECT_ROOT") else parent_path

        if parent_path is not None:
            parent_path = str(Path(parent_path).absolute())

            for key, path in self.__dict__.items():
                if isinstance(path, str) and not Path(path).is_absolute():
                    if for_logging and key != "filename":
                        # In the case of logging configs, apply only on filename
                        continue
                    self.__dict__.update({key: str(Path(parent_path).joinpath(path))})

                elif isinstance(path, CONFIG):
                    path.update_paths(parent_path=parent_path, for_logging=for_logging)

        else:
            print("Paths can't be converted to absolute paths, since no PROJECT_ROOT is found!")


def _set_wd() -> None:
    """Set the given directory as new working directory of the project."""
    if PROJECT_NAME not in str(Path.cwd()):
        msg = f'Current working dir "{Path.cwd()}" is outside of project "{PROJECT_NAME}".'
        raise ValueError(msg)

    print("\033[94m" + f"Current working dir:\t{Path.cwd()}" + "\033[0m")  # blue print

    new_dir = Path(PROJECT_ROOT)
    if new_dir == Path.cwd():
        return

    if new_dir.exists():
        os.chdir(new_dir)
        print("\033[93m" + f"New working dir:\t{Path.cwd()}\n" + "\033[0m")  # yellow print
    else:
        print("\033[91m" + f"Given folder not found. Working dir remains:\t{Path.cwd()}\n" + "\033[0m")  # red print


def update_logger_configs(
    new_logger_name: str, new_logger_filename: str | Path, logger: logging.Logger
) -> logging.Logger:
    """
    Update logger name and filename.

    :param new_logger_name: new logger name
    :param new_logger_filename: new logger filename
    :param logger: updated logger object
    """
    # Set new logger name
    logger.name = new_logger_name

    # Check filename
    if not str(new_logger_filename).endswith(".log"):
        msg = f"Given filename '{new_logger_filename}' does not end with '.log'."
        raise ValueError(msg)

    # Create parent dirs
    Path(new_logger_filename).parent.mkdir(parents=True, exist_ok=True)

    # Overwrite logger
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            new_file_handler = logging.FileHandler(filename=new_logger_filename)
            new_file_handler.setFormatter(handler.formatter)
            new_file_handler.setLevel(handler.level)
            logger.removeHandler(handler)
            logger.addHandler(new_file_handler)
    return logger


# %% Setup configuration object < o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Instantiate config object
config = CONFIG()

# Load config file(s)
for config_file in Path(__file__).parent.glob("../configs/*config.toml"):
    config.update(new_configs=toml.load(str(config_file)))

# Extract some useful globals
PROJECT_NAME = config.PROJECT_NAME
PROJECT_ROOT = __file__[: __file__.find(PROJECT_NAME) + len(PROJECT_NAME)]

# Set root path to config file & update paths
config.paths.PROJECT_ROOT = PROJECT_ROOT
config.paths.update_paths()

# Prepare logging
config.logging.update_paths(parent_path=PROJECT_ROOT, for_logging=True)
_create_parent_dirs(config_as_dict=config.logging.asdict())

# Extract paths and variables, and set logging configs
paths = config.paths
params = config.params
logging.config.dictConfig(config.logging.asdict())

# Welcome
print(
    "\n"
    + ("*" * 95 + "\n") * 2
    + "\n"
    + "\t" * 10
    + config.PROJECT_ICON
    + PROJECT_NAME
    + " "
    + config.PROJECT_ICON
    + "\n" * 2
    + ("*" * 95 + "\n") * 2
)

# Set project working directory
_set_wd()

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
