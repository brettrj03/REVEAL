"""Helpers for loading the project's environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Union

from dotenv import load_dotenv

PathLike = Union[str, Path]


@lru_cache(maxsize=1)
def load_project_env(dotenv_path: PathLike | None = None) -> bool:
    """Load environment variables from the project's .env file once.

    Parameters
    ----------
    dotenv_path: Optional explicit path to a .env file. When omitted we
        automatically resolve to the repository root .env file.

    Returns
    -------
    bool
        True if the file existed and variables were loaded, False otherwise.
    """

    if dotenv_path is None:
        project_root = Path(__file__).resolve().parents[2]
        dotenv_path = project_root / ".env"
    else:
        dotenv_path = Path(dotenv_path)

    # load_dotenv quietly returns False when the file is missing, which is fine.
    return load_dotenv(dotenv_path, override=False)
