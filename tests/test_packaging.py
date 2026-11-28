from importlib import resources


def test_package_ships_py_typed_marker() -> None:
    """PEP 561: downstream type checkers only use our annotations with this marker."""
    assert resources.files("evalforge").joinpath("py.typed").is_file()
