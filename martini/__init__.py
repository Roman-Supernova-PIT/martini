from importlib.metadata import version, PackageNotFoundError
from .host_association import compute_offset_angle, compute_dlr, find_host

__all__ = ['compute_offset_angle', 'compute_dlr', 'find_host']


try:
    __version__ = version("martini")
except PackageNotFoundError:
    # package is not installed
    pass
