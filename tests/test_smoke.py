"""Smoke tests — verify the package imports and version is set."""

import hybrid_arch


def test_version_exists():
    assert hasattr(hybrid_arch, "__version__")


def test_version_is_string():
    assert isinstance(hybrid_arch.__version__, str)


def test_version_not_empty():
    assert hybrid_arch.__version__ != ""
