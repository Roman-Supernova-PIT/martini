"""Transient host association using the Directional Light Radius (DLR) method.

Candidate catalog columns
-------------------------
All methods require:

* ``ra``, ``dec`` -- galaxy centroid in degrees
* ``a``           -- semi-major axis in pixels (converted with ``pixel_scale``)

``method="apa"`` additionally requires:

* ``b``     -- semi-minor axis in pixels
* ``theta`` -- major-axis position angle in degrees, image frame
  (``theta = 90`` points the major axis North)

``method="ge"`` additionally requires the ellipse coefficients
``cxx``, ``cyy``, ``cxy`` (SExtractor convention).

GE axis-ratio conventions
-------------------------
For the GE method the ellipticity ``e = sqrt(Q**2 + U**2)`` is derived from the
ellipse coefficients, with ``e = (a**2 - b**2) / (a**2 + b**2)``.

* ``"exact"`` -- ``a/b = sqrt((1 + e) / (1 - e))``, the true axis ratio.
* ``"ghost"`` -- ``a/b = (1 + e) / (1 - e)``, the convention used by GHOST,
  which equals the square of the true axis ratio.

References
----------
Sullivan et al. (2006), ApJ, 648, 868.
Gupta et al. (2016), AJ, 152, 154.
"""

__all__ = ['compute_ddlr', 'ROMAN_PIXEL_SCALE']

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

ROMAN_PIXEL_SCALE = 1


def compute_ddlr(sn_ra, sn_dec, cat, method="apa",
                 pixel_scale=ROMAN_PIXEL_SCALE, ge_axis_ratio="ghost",
                 ddlr_threshold=None):
    """Compute d_DLR for every candidate galaxy and sort by it.

    Parameters
    ----------
    sn_ra, sn_dec : float
        SN coordinates in degrees.
    cat : pandas.DataFrame
        Candidate galaxies with the columns listed above.
    method : {"apa", "ge"}
        "apa" uses a, b, theta.  "ge" uses the ellipse coefficients.
    pixel_scale : float
        Arcsec per pixel.
    ge_axis_ratio : {"ghost", "exact"}
        Axis-ratio convention for the GE method (see module docstring).
    ddlr_threshold : float, optional
        If given, drop candidates with ``ddlr >= ddlr_threshold``.  A value of
        4.0 follows Sullivan et al. (2006).  Default ``None`` keeps all
        candidates.

    Returns
    -------
    pandas.DataFrame
        Copy of ``cat`` with ``sep`` (arcsec), ``dlr`` (arcsec) and ``ddlr``,
        sorted so the first row is the preferred host.  May be empty if
        ``ddlr_threshold`` excludes every candidate.
    """
    df = cat.copy()

    if method == "apa":
        phi_image = df["theta"].values
        a_over_b = df["a"].values / df["b"].values
    elif method == "ge":
        cxx, cyy, cxy = df["cxx"].values, df["cyy"].values, df["cxy"].values
        Q = (cyy - cxx) / (cxx + cyy)
        U = -cxy / (cxx + cyy)
        phi_image = np.degrees(0.5 * np.arctan2(U, Q))
        e = np.clip(np.hypot(Q, U), 0, 1 - 1e-12)
        if ge_axis_ratio == "ghost":
            a_over_b = (1 + e) / (1 - e)
        elif ge_axis_ratio == "exact":
            a_over_b = np.sqrt((1 + e) / (1 - e))
        else:
            raise ValueError("ge_axis_ratio must be 'ghost' or 'exact'")
    else:
        raise ValueError("method must be 'apa' or 'ge'")

    # Major axis as a sky position angle (east of north)
    phi = np.radians(phi_image - 90.0)

    # Separation and bearing from each galaxy to the SN (handles cos(dec) and quadrants)
    sn = SkyCoord(sn_ra, sn_dec, unit="deg")
    gal = SkyCoord(df["ra"].values, df["dec"].values, unit="deg")
    df["sep"] = gal.separation(sn).to_value(u.arcsec)
    psi = gal.position_angle(sn).to_value(u.rad)

    # DLR along the SN direction (Sullivan et al. 2006)
    eta = phi - psi
    a = df["a"].values * pixel_scale
    df["dlr"] = a / np.sqrt((a_over_b * np.sin(eta)) ** 2 + np.cos(eta) ** 2)
    df["ddlr"] = df["sep"] / df["dlr"]

    if ddlr_threshold is not None:
        df = df[df["ddlr"] < ddlr_threshold]

    return df.sort_values("ddlr").reset_index(drop=True)
