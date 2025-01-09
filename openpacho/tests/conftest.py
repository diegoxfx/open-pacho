import pytest

def pytest_addoption(parser):
    parser.addoption("--device-type", action="store", default="GPU")
    parser.addoption("--device-index", action="store", default=0)

@pytest.fixture(scope='session')
def device_type(request):
    return request.config.option.device_type


@pytest.fixture(scope='session')
def device_index(request):
    return request.config.option.device_index