"""Sanity check that the installable package imports."""

from claimforge import __version__


def test_version() -> None:
    assert __version__ == "0.1.0"
