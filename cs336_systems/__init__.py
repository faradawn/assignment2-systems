# cs336_systems/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cs336-systems")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback when running from source
