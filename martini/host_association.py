"""Transient host association using the Directional Light Radius (DLR) method."""

__all__ = ['compute_offset_angle', 'compute_dlr', 'find_host']

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def compute_offset_angle(sn_ra, sn_dec, candidate_hosts):
    """
    Compute the offset angle (gamma) between the transient and each candidate host.

    Gamma is the angle between the galaxy's major axis and the direction from the
    galaxy center to the transient, measured in radians.

    Parameters
    ----------
    sn_ra : float
        RA of the transient in degrees.
    sn_dec : float
        Dec of the transient in degrees.
    candidate_hosts : `~pandas.DataFrame`
        DataFrame of candidate host galaxies.  Must contain columns:
        * ``ra``    galaxy RA in degrees
        * ``dec``   galaxy Dec in degrees
        * ``theta`` position angle in degrees

    Returns
    -------
    result : `~pandas.DataFrame`
        Copy of ``candidate_hosts`` with a new ``gamma`` column (radians).
    """
    xr = np.abs(sn_ra - candidate_hosts['ra'].values)
    yr = np.abs(sn_dec - candidate_hosts['dec'].values)

    phi = np.deg2rad(candidate_hosts['theta'].values)

    result = candidate_hosts.copy()
    result['gamma'] = phi - np.arctan(yr / xr)
    return result


def compute_dlr(candidate_hosts):
    """
    Compute the Directional Light Radius (DLR) for each candidate host galaxy.

    Parameters
    ----------
    candidate_hosts : `~pandas.DataFrame`
        DataFrame of candidate host galaxies.  Must contain columns:
        * ``a``     semi-major axis in arcseconds
        * ``b``     semi-minor axis in arcseconds
        * ``gamma`` offset angle in radians (output of `compute_offset_angle`)

    Returns
    -------
    result : `~pandas.DataFrame`
        Copy of ``candidate_hosts`` with a new ``dlr`` column (arcseconds).


    References
    ----------
    Sullivan et al. (2006), ApJ, 648, 868.
    Gupta et al. (2016), AJ, 152, 154.
    """
    a = candidate_hosts['a'].values
    b = candidate_hosts['b'].values
    gamma = candidate_hosts['gamma'].values

    dlr = (a * b) / np.sqrt((a * np.sin(gamma)) ** 2 + (b * np.cos(gamma)) ** 2)

    result = candidate_hosts.copy()
    result['dlr'] = dlr
    return result


def find_host(sn_ra, sn_dec, candidate_hosts, ddlr_threshold=4.0):
    """
    Identify potential host galaxies matchs for a transient using the DLR method.

    Parameters
    ----------
    sn_ra : float
        RA of the transient in degrees.
    sn_dec : float
        Dec of the transient in degrees.
    candidate_hosts : `~pandas.DataFrame`
        DataFrame of candidate host galaxies.  Must contain columns:
        ``ra``, ``dec``, ``a``, ``b``, ``theta``.
    ddlr_threshold : float, optional
        Maximum delta_DLR to consider a galaxy a potential host.  Default is 4.0,
        following the convention of Sullivan et al. (2006).

    Returns
    -------
    all_hosts : `~pandas.DataFrame`
        Full DataFrame with added columns ``gamma`` (rad), ``dlr`` (arcsec),
        ``sep`` (arcsec), and ``ddlr`` (dimensionless), sorted by ``ddlr``.
    potential_hosts : `~pandas.DataFrame`
        Subset of ``all_hosts`` with ``ddlr < ddlr_threshold``.
    """
    gdf = compute_offset_angle(sn_ra, sn_dec, candidate_hosts)
    gdf = compute_dlr(gdf)

    sn_coord = SkyCoord(sn_ra, sn_dec, unit='deg')
    gal_coords = SkyCoord(gdf['ra'].values, gdf['dec'].values, unit='deg')
    gdf['sep'] = sn_coord.separation(gal_coords).arcsec
    gdf['ddlr'] = gdf['sep'] / gdf['dlr']

    gdf = gdf.sort_values('ddlr').reset_index(drop=True)
    potential_hosts = gdf[gdf['ddlr'] < ddlr_threshold].reset_index(drop=True)

    return gdf, potential_hosts
