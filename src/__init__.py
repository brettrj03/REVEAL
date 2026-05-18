"""Project package initialisation."""

from .utils.env import load_project_env

# Ensure environment variables from .env are available once src is imported.
load_project_env()
