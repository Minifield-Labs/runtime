import pytest

from minifield_converters.fixture import write_fixture


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    write_fixture(path)
    return path
