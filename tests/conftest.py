import subprocess

import pytest

from lightonocr_math.render import IMAGE


@pytest.fixture(scope="session")
def docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "image", "inspect", IMAGE], check=False, capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def pytest_collection_modifyitems(config, items):
    docker_items = [i for i in items if i.get_closest_marker("docker")]
    if not docker_items:
        return
    try:
        ok = subprocess.run(["docker", "image", "inspect", IMAGE], check=False, capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    if not ok:
        skip = pytest.mark.skip(reason="docker or texlive image unavailable")
        for i in docker_items:
            i.add_marker(skip)
