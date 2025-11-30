import os

import pytest
from pyspark.sql import SparkSession

current_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the path to the config.json file
config_path = os.path.join(current_dir, "..", "config", "dev.json")


@pytest.fixture(scope="session")
def pkg_path():
    return os.path.join(current_dir, "..", "walmart_ahmedkobtan_agentic_store_operations")

@pytest.fixture(scope="session")
def data_path():
    return os.path.join(current_dir, "..", "walmart_ahmedkobtan_agentic_store_operations", "data")
