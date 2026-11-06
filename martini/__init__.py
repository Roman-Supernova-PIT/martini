from importlib.metadata import version, PackageNotFoundError
from .host_association import compute_ddlr, ROMAN_PIXEL_SCALE

__all__ = ['compute_ddlr', 'ROMAN_PIXEL_SCALE']


try:
    __version__ = version("martini")
except PackageNotFoundError:
    # package is not installed
    pass
