"""Command-line interface for martini host association tools."""

from .host_association import compute_dlr, find_host  # noqa: F401

__all__ = ['main']


def main(args=None):
    """Entry point for the ``martini`` command-line tool."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Transient host association using the DLR method.'
    )
    parser.add_argument('transient_ra', type=float, help='Transient RA in degrees.')
    parser.add_argument('transient_dec', type=float, help='Transient Dec in degrees.')

    res = parser.parse_args(args)

    print(f'Transient position: RA={res.transient_ra:.6f} deg, Dec={res.transient_dec:.6f} deg')
    print('Provide a galaxy catalog to find_host() to identify the most likely host.')
