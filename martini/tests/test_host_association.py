import numpy as np
import pandas as pd
import pytest

from ..host_association import compute_ddlr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_catalog(**kwargs):
    """Build a single-row DataFrame with default values overridden by kwargs."""
    defaults = {'ra': 10.0, 'dec': 0.0, 'a': 5.0, 'b': 3.0, 'theta': 90.0}
    defaults.update(kwargs)
    return pd.DataFrame({k: [v] for k, v in defaults.items()})


def add_ellipse_coefficients(cat):
    """Add SExtractor-style cxx, cyy, cxy columns computed from a, b, theta."""
    df = cat.copy()
    t = np.radians(df['theta'])
    a2, b2 = df['a'] ** 2, df['b'] ** 2
    df['cxx'] = np.cos(t) ** 2 / a2 + np.sin(t) ** 2 / b2
    df['cyy'] = np.sin(t) ** 2 / a2 + np.cos(t) ** 2 / b2
    df['cxy'] = 2 * np.cos(t) * np.sin(t) * (1 / a2 - 1 / b2)
    return df


@pytest.fixture
def two_galaxy_catalog():
    """Two galaxies; the first is much closer to the transient at (10.0, 0.0)."""
    return pd.DataFrame({
        'ra':    [10.5,  10.0],
        'dec':   [0.0,   0.001],
        'a':     [5.0,   5.0],
        'b':     [3.0,   3.0],
        'theta': [45.0,  45.0],
    })


# ---------------------------------------------------------------------------
# General behaviour
# ---------------------------------------------------------------------------

class TestComputeDdlrGeneral:

    @pytest.mark.parametrize('method', ['apa', 'ge'])
    def test_adds_required_columns(self, two_galaxy_catalog, method):
        cat = add_ellipse_coefficients(two_galaxy_catalog)
        result = compute_ddlr(10.0, 0.0, cat, method=method)
        for col in ('sep', 'dlr', 'ddlr'):
            assert col in result.columns

    def test_length_matches_catalog(self, two_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, two_galaxy_catalog)
        assert len(result) == len(two_galaxy_catalog)

    def test_sorted_by_ddlr(self, two_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, two_galaxy_catalog)
        assert list(result['ddlr']) == sorted(result['ddlr'])
        assert list(result.index) == list(range(len(result)))

    def test_closest_galaxy_first(self, two_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, two_galaxy_catalog)
        assert np.isclose(result['dec'].values[0], 0.001)

    def test_does_not_mutate_input(self, two_galaxy_catalog):
        original = two_galaxy_catalog.copy()
        _ = compute_ddlr(10.0, 0.0, two_galaxy_catalog)
        pd.testing.assert_frame_equal(two_galaxy_catalog, original)

    def test_sep_in_arcsec(self):
        cat = make_catalog(ra=10.0, dec=0.0)
        result = compute_ddlr(10.0, 1.0 / 3600, cat)
        assert np.isclose(result['sep'].values[0], 1.0, rtol=1e-6)

    def test_ddlr_equals_sep_over_dlr(self, two_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, two_galaxy_catalog)
        expected = result['sep'] / result['dlr']
        assert np.allclose(result['ddlr'].values, expected.values, rtol=1e-10)

    def test_invalid_method_raises(self, two_galaxy_catalog):
        with pytest.raises(ValueError, match='method'):
            compute_ddlr(10.0, 0.0, two_galaxy_catalog, method='bogus')

    def test_invalid_ge_axis_ratio_raises(self, two_galaxy_catalog):
        cat = add_ellipse_coefficients(two_galaxy_catalog)
        with pytest.raises(ValueError, match='ge_axis_ratio'):
            compute_ddlr(10.0, 0.0, cat, method='ge', ge_axis_ratio='bogus')


# ---------------------------------------------------------------------------
# APA method geometry
# ---------------------------------------------------------------------------

class TestComputeDdlrApa:

    def test_sn_along_major_axis_gives_a(self):
        """theta = 90 → major axis points North; SN due North → DLR = a."""
        a, b = 5.0, 3.0
        cat = make_catalog(a=a, b=b, theta=90.0)
        result = compute_ddlr(10.0, 0.001, cat)
        assert np.isclose(result['dlr'].values[0], a, rtol=1e-6)

    def test_sn_along_minor_axis_gives_b(self):
        """theta = 90 → major axis North; SN due East → DLR = b."""
        a, b = 5.0, 3.0
        cat = make_catalog(a=a, b=b, theta=90.0)
        result = compute_ddlr(10.001, 0.0, cat)
        assert np.isclose(result['dlr'].values[0], b, rtol=1e-6)

    def test_theta_zero_major_axis_east_west(self):
        """theta = 0 → major axis East-West; SN due East → DLR = a."""
        a, b = 5.0, 3.0
        cat = make_catalog(a=a, b=b, theta=0.0)
        result = compute_ddlr(10.001, 0.0, cat)
        assert np.isclose(result['dlr'].values[0], a, rtol=1e-6)

    def test_dlr_bounded_by_axes(self):
        a, b = 6.0, 2.0
        rng = np.random.default_rng(0)
        cat = make_catalog(a=a, b=b, theta=30.0)
        for _ in range(20):
            sn_ra = 10.0 + rng.uniform(-0.01, 0.01)
            sn_dec = rng.uniform(-0.01, 0.01)
            dlr = compute_ddlr(sn_ra, sn_dec, cat)['dlr'].values[0]
            assert b - 1e-10 <= dlr <= a + 1e-10

    def test_circular_galaxy_constant_dlr(self):
        r = 4.0
        for theta in (0.0, 30.0, 90.0, 137.0):
            cat = make_catalog(a=r, b=r, theta=theta)
            result = compute_ddlr(10.003, 0.002, cat)
            assert np.isclose(result['dlr'].values[0], r, rtol=1e-10)

    def test_pixel_scale_scales_dlr(self):
        cat = make_catalog(theta=45.0)
        r1 = compute_ddlr(10.001, 0.002, cat, pixel_scale=1.0)
        r2 = compute_ddlr(10.001, 0.002, cat, pixel_scale=0.11)
        assert np.isclose(r2['dlr'].values[0], 0.11 * r1['dlr'].values[0], rtol=1e-10)
        assert np.isclose(r2['sep'].values[0], r1['sep'].values[0], rtol=1e-10)


# ---------------------------------------------------------------------------
# GE method
# ---------------------------------------------------------------------------

class TestComputeDdlrGe:

    @pytest.mark.parametrize('theta', [0.0, 30.0, 90.0, 137.0])
    def test_exact_matches_apa(self, theta):
        """With coefficients built from a, b, theta, GE 'exact' reproduces APA."""
        cat = add_ellipse_coefficients(make_catalog(a=6.0, b=2.0, theta=theta))
        apa = compute_ddlr(10.002, 0.001, cat, method='apa')
        ge = compute_ddlr(10.002, 0.001, cat, method='ge', ge_axis_ratio='exact')
        assert np.isclose(ge['dlr'].values[0], apa['dlr'].values[0], rtol=1e-8)

    def test_ghost_uses_squared_axis_ratio(self):
        """'ghost' convention gives a/b = (a_true/b_true)**2; along minor axis DLR = b**2/a."""
        a, b = 6.0, 2.0
        cat = add_ellipse_coefficients(make_catalog(a=a, b=b, theta=90.0))
        result = compute_ddlr(10.001, 0.0, cat, method='ge', ge_axis_ratio='ghost')
        assert np.isclose(result['dlr'].values[0], b ** 2 / a, rtol=1e-6)

    def test_ghost_and_exact_agree_along_major_axis(self):
        cat = add_ellipse_coefficients(make_catalog(a=6.0, b=2.0, theta=90.0))
        ghost = compute_ddlr(10.0, 0.001, cat, method='ge', ge_axis_ratio='ghost')
        exact = compute_ddlr(10.0, 0.001, cat, method='ge', ge_axis_ratio='exact')
        assert np.isclose(ghost['dlr'].values[0], exact['dlr'].values[0], rtol=1e-6)
        assert np.isclose(exact['dlr'].values[0], 6.0, rtol=1e-6)


# ---------------------------------------------------------------------------
# ddlr_threshold
# ---------------------------------------------------------------------------

@pytest.fixture
def three_galaxy_catalog():
    """Three galaxies at increasing distance North of the transient at (10.0, 0.0)."""
    return pd.DataFrame({
        'ra':    [10.0,  10.0, 10.0],
        'dec':   [0.001, 0.01, 0.1],
        'a':     [5.0,   5.0,  5.0],
        'b':     [3.0,   3.0,  3.0],
        'theta': [45.0,  45.0, 45.0],
    })


class TestComputeDdlrThreshold:

    def test_default_keeps_all_candidates(self, three_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, three_galaxy_catalog)
        assert len(result) == len(three_galaxy_catalog)

    def test_drops_candidates_at_or_above_cut(self, three_galaxy_catalog):
        full = compute_ddlr(10.0, 0.0, three_galaxy_catalog)
        cut = full['ddlr'].values[1]  # exactly the second candidate's ddlr
        result = compute_ddlr(10.0, 0.0, three_galaxy_catalog, ddlr_threshold=cut)
        assert len(result) == 1
        assert (result['ddlr'] < cut).all()

    def test_result_is_sorted_and_reindexed(self, three_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, three_galaxy_catalog, ddlr_threshold=1e6)
        assert list(result['ddlr']) == sorted(result['ddlr'])
        assert list(result.index) == list(range(len(result)))

    def test_zero_threshold_returns_empty(self, three_galaxy_catalog):
        result = compute_ddlr(10.0, 0.0, three_galaxy_catalog, ddlr_threshold=0.0)
        assert len(result) == 0
        for col in ('sep', 'dlr', 'ddlr'):
            assert col in result.columns

    def test_tighter_threshold_keeps_fewer(self, three_galaxy_catalog):
        full = compute_ddlr(10.0, 0.0, three_galaxy_catalog)
        mid = full['ddlr'].values[1:].mean()
        tight = compute_ddlr(10.0, 0.0, three_galaxy_catalog, ddlr_threshold=mid)
        loose = compute_ddlr(10.0, 0.0, three_galaxy_catalog, ddlr_threshold=1e6)
        assert len(tight) < len(loose)
