#!/usr/bin/python3
import pytest
import utils


class TestPTS:
    """This is an PTS Test Suite Class to test Mathlibs component of TheRock"""

    @pytest.mark.parametrize(
        argnames=("name"),
        argvalues=(
            "benchmark_rocrand_host_api",
            "benchmark_rocrand_device_api",
        ),
    )
    def test_rocrand(self, name, rockDir):
        """A Test case to run rocrand benchmark"""
        ret, out = utils.runCmdGetOutput(
            f"./{name}",
            "--trials",
            "1000",
            cwd=f"{rockDir}/bin",
        )
        assert ret == 0
