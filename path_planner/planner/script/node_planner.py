#!/usr/bin/env python3

from utils.config import load_yaml_file
from utils import constants
from utils.planner import PathPlanner
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting the planning node...")
    
    # Load the config
    config = load_yaml_file(constants.merged_config_path, __file__)

    # Initialize the planner class
    planner = PathPlanner(
        config=config
    )

    # Run the planner
    planner.run()