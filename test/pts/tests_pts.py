#!/usr/bin/python3
import re
import pytest
import utils


class TestPTS:
    """This is an PTS Test Suite Class to test Mathlibs component of TheRock"""

    @pytest.mark.parametrize(
        argnames=("benchmark"),
        argvalues=(
            "benchmark_rocrand_host_api",
            "benchmark_rocrand_device_api",
        ),
    )
    def test_rocrand(self, benchmark, rockDir, dbDoc):
        """A Test case to run rocrand benchmark"""
        ret, out = utils.runCmdGetOutput(
            f"./{benchmark}",
            "--trials",
            "1000",
            cwd=f"{rockDir}/bin",
        )
        assert ret == 0
        if not dbDoc:
            return
        dbDoc.update(
            {
                "_index": "pts_rocrand_benchmark_data-v1.0.0",
                "executable": benchmark,
                "scores": [],
            }
        )
        expr = r"(?P<name>.*?/\w+)"
        expr += r" +(?P<real_time>\d+) (?P<time_unit>\w+)"
        expr += r" +(?P<cpu_time>\d+) \w+"
        expr += r" +(?P<iterations>\d+)"
        expr += r" +(?P<bps>[\d\.]+)(?P<bpsMul>[KMGT])i/s"
        expr += r" +(?P<ips>[\d\.]+)(?P<ipsMul>[KMGT])/s"
        for mtch in re.finditer(expr, out):
            scoreCard = mtch.groupdict()
            # spliting benchmark name
            nameSplit = re.search(r"<(.*?)[\(>]", scoreCard["name"]).group(1).split(",")
            scoreCard["engine"] = nameSplit[0]
            if len(nameSplit) == 3:
                scoreCard["mode"] = nameSplit[1]
            scoreCard["distribution"] = nameSplit[-1]
            # score multiplier
            scoreCard["bytes_per_second"] = utils.normBps(
                scoreCard.pop("bps"), scoreCard.pop("bpsMul")
            )
            scoreCard["items_per_second"] = utils.normIps(
                scoreCard.pop("ips"), scoreCard.pop("ipsMul")
            )
            dbDoc["scores"].append(scoreCard)
