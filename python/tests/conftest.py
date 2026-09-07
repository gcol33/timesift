import pytest

from timesift.registry import RESPONSES


@pytest.fixture
def temporary_response():
    """Register a response head for one test and take it out again afterwards, so the registry
    the next test sees is the one that ships."""
    names = []

    def register(name, spec):
        from timesift import register_response
        names.append(name)
        return register_response(name, spec, overwrite=True)

    yield register
    for name in names:
        RESPONSES.remove(name)
