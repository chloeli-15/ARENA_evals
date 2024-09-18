import pathlib


def get_workspace_dir() -> pathlib.Path:

    current_dir = pathlib.Path(__file__).parent.resolve()

    return current_dir


def get_workspace_filepath_for_filename(filename: str) -> pathlib.Path:

    return get_workspace_dir() / filename
