import os
import logging
import pytest
import elasticsearch

# disable SSL warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.SecurityWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import utils
logging.getLogger("urllib3").setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def pytest_addoption(parser):
    """Initialization of cmdline args"""
    parser.addoption("--rock-dir", action="store", help="Path of TheRock Dir")
    parser.addoption("--push-db", action="store_true", help="Enable to push results to DB")


@pytest.fixture(scope="session")
def rockDir(pytestconfig):
    """Fixture to access the path to the TheRock passed by cmdline arg: --rock-dir"""
    return pytestconfig.getoption("--rock-dir")


@pytest.fixture(scope="session")
def dbSession(pytestconfig):
    if not pytestconfig.getoption("--push-db"):
        yield None
        return
    # these must be set in GitHub Actions or local environment
    assert (PTS_DB_API_ID := os.environ.get("PTS_DB_API_ID"))
    assert (PTS_DB_API_KEY := os.environ.get("PTS_DB_API_KEY"))
    # establish Elasticsearch Connection
    host = "elasticdev.amd.com"
    port = 9200
    session = elasticsearch.Elasticsearch(
        f"https://{host}:{port}",
        verify_certs=False,
        api_key=(PTS_DB_API_ID, PTS_DB_API_KEY),
        ssl_show_warn=False,
        max_retries=10,
        retry_on_timeout=True,
    )
    # simple health check
    assert session.ping(), "Elasticsearch ping failed!"
    log.info("Connected to Elasticsearch")
    yield session
    session and session.transport.close()


@pytest.fixture(scope="function")
def dbIngress(dbSession):
    if not dbSession:
        yield None
        return
    ingress = {}
    yield ingress
    resp = elasticsearch.helpers.bulk(dbSession, (ingress, ), chunk_size=100, request_timeout=60*30)
    assert resp, 'DB Ingestion Failed: {resp}'
