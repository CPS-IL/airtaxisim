import json
import os
import random
import shutil
import sys
import yaml
import numpy as np

from datetime import datetime
from loguru import logger as log

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import constants
from utils.logging import set_logger

def load_yaml_file(file_path, file_name=None):
    if not os.path.exists(file_path):
        log.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    log.trace(f"Loading YAML file: {file_path}")
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    try:
        set_logger(config['loglevel'], file_name)
    except KeyError:
        pass

    return config

def write_shared_tmp_file(file_name, data):
    directory = os.path.dirname(constants.merged_config_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, file_name), 'w', buffering=1) as file:
        if ".json" in file_name:
            json.dump(data, file, indent=4)
        else:
            file.write(str(data))
        file.flush()

def read_shared_tmp_file(file_name):
    directory = os.path.dirname(constants.merged_config_path)
    with open(os.path.join(directory, file_name), 'r') as file:
        if ".json" in file_name:
            content = json.load(file)
        else:
            content = file.read()
    return content

def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1:
            dict1[key] = deep_merge(dict1.get(key, {}), value)
        else:
            dict1[key] = value
    return dict1


def parse_includes(config, base_directory):
    for key, value in config.items():
        if isinstance(value, dict):
            if 'include' in value:
                include_file = value['include']
                include_path = os.path.join(base_directory, include_file)
                included_config = load_yaml_file(include_path)
                log.info(f"Including file {include_file} into field {key}")
                config[key] = deep_merge(config[key], included_config)
            else:
                config[key] = parse_includes(value, base_directory)
    return config

def parse_hierarchical_config(config_file):
    log.info(f"Loading config from: {config_file}")
    config = load_yaml_file(config_file)
    return parse_includes(config, os.path.dirname(config_file))

def write_flattened_config(file_path, config):

    file_path = os.path.abspath(file_path)
    base_directory = os.path.dirname(file_path)
    timestamp_file = os.path.join(base_directory, constants.timestamp_file)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    if os.path.exists(base_directory) and os.path.isdir(base_directory):
        if "record_persistent" not in config:
            config["record_persistent"] = False
        if config["record_persistent"] and os.path.exists(timestamp_file):
            with open(timestamp_file, "r") as file:
                timestamp_record = file.read().strip()
            record_dir = os.path.join(constants.records_dir, timestamp_record)
            os.makedirs(record_dir, exist_ok=True)
            shutil.move(base_directory, record_dir)
        else:
            shutil.rmtree(base_directory)

    os.makedirs(base_directory)
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
    with open(timestamp_file, 'w') as file:
        file.write(timestamp)

    print(f"Config successfully written to {file_path}")