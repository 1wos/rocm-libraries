import logging
import pytest

import utils

logging.getLogger("urllib3").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Initialization of cmdline args"""
    parser.addoption("--rock-dir", action="store", help="Path of TheRock Dir")


@pytest.fixture(scope="session")
def rockDir(pytestconfig):
    """Fixture to access the path to the TheRock passed by cmdline arg: --rock-dir"""
    rockDir = pytestconfig.getoption("--rock-dir")
    return rockDir
