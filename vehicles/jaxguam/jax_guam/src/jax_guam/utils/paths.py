import pathlib


def package_dir() -> pathlib.Path:
    path = pathlib.Path(__file__).parent.parent
    assert path.name == "jax_guam"
    assert path.exists()
    return path


def data_dir() -> pathlib.Path:
    path = package_dir() / "data"
    assert path.exists()
    return path
