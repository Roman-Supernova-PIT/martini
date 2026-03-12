import numpy as np
import pandas as pd
import pytest

from ..host_association import compute_offset_angle, compute_dlr, find_host


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_catalog(**kwargs):
    """Build a single-row DataFrame with default values overridden by kwargs."""
    defaults = {'ra': 10.0, 'dec': 0.0, 'a': 5.0, 'b': 3.0, 'theta': 0.0}
    defaults.update(kwargs)
    return pd.DataFrame({k: [v] for k, v in defaults.items()})


# ---------------------------------------------------------------------------
# compute_offset_angle tests
# ---------------------------------------------------------------------------

class TestComputeOffsetAngle:

    def test_adds_gamma_column(self):
        cat = make_catalog(ra=10.0, dec=0.0, theta=45.0)
        result = compute_offset_angle(11.0, 1.0, cat)
        assert 'gamma' in result.columns

    def test_gamma_value(self):
        """gamma = theta_rad - arctan(|Δdec| / |Δra|)."""
        # xr = |11 - 10| = 1, yr = |1 - 0| = 1 → arctan(1) = π/4
        # theta = 45 deg → phi = π/4 rad → gamma = π/4 - π/4 = 0
        cat = make_catalog(ra=10.0, dec=0.0, theta=45.0)
        result = compute_offset_angle(11.0, 1.0, cat)
        assert np.isclose(result['gamma'].values[0], 0.0, atol=1e-10)

    def test_does_not_mutate_input(self):
        cat = make_catalog(ra=10.0, dec=0.0, theta=45.0)
        original_cols = set(cat.columns)
        _ = compute_offset_angle(11.0, 1.0, cat)
        assert set(cat.columns) == original_cols

    def test_multiple_galaxies(self):
        cat = pd.DataFrame({
            'ra':    [10.0, 10.0],
            'dec':   [0.0,  0.0],
            'a':     [5.0,  5.0],
            'b':     [3.0,  3.0],
            'theta': [45.0, 90.0],
        })
        result = compute_offset_angle(11.0, 1.0, cat)
        assert len(result) == 2
        # Row 0: gamma = π/4 - π/4 = 0
        assert np.isclose(result['gamma'].values[0], 0.0, atol=1e-10)
        # Row 1: gamma = π/2 - π/4 = π/4
        assert np.isclose(result['gamma'].values[1], np.pi / 4, atol=1e-10)

    def test_north_south_alignment(self):
        """When Δra = 0, arctan(yr/0) → π/2 (numpy handles inf correctly)."""
        cat = make_catalog(ra=10.0, dec=0.0, theta=0.0)
        result = compute_offset_angle(10.0, 1.0, cat)  # directly North
        # arctan(inf) = π/2 → gamma = 0 - π/2 = -π/2
        assert np.isclose(result['gamma'].values[0], -np.pi / 2, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_dlr tests
# ---------------------------------------------------------------------------

class TestComputeDlr:

    def _make_with_gamma(self, a, b, gamma_rad):
        return pd.DataFrame({'a': [a], 'b': [b], 'gamma': [gamma_rad]})

    def test_adds_dlr_column(self):
        df = self._make_with_gamma(5.0, 3.0, 0.0)
        result = compute_dlr(df)
        assert 'dlr' in result.columns

    def test_along_major_axis_gives_a(self):
        """gamma = 0 → DLR = a*b / sqrt(0 + b^2) = a."""
        a, b = 5.0, 3.0
        result = compute_dlr(self._make_with_gamma(a, b, 0.0))
        assert np.isclose(result['dlr'].values[0], a, rtol=1e-10)

    def test_along_minor_axis_gives_b(self):
        """gamma = π/2 → DLR = a*b / sqrt(a^2 + 0) = b."""
        a, b = 5.0, 3.0
        result = compute_dlr(self._make_with_gamma(a, b, np.pi / 2))
        assert np.isclose(result['dlr'].values[0], b, rtol=1e-10)

    def test_dlr_bounded_by_axes(self):
        """DLR must always be between b and a (inclusive)."""
        a, b = 6.0, 2.0
        rng = np.random.default_rng(0)
        gammas = rng.uniform(0, 2 * np.pi, 50)
        df = pd.DataFrame({'a': a, 'b': b, 'gamma': gammas})
        result = compute_dlr(df)
        assert (result['dlr'].values >= b - 1e-10).all()
        assert (result['dlr'].values <= a + 1e-10).all()

    def test_does_not_mutate_input(self):
        df = self._make_with_gamma(5.0, 3.0, 0.5)
        original_cols = set(df.columns)
        _ = compute_dlr(df)
        assert set(df.columns) == original_cols

    def test_circular_galaxy_constant_dlr(self):
        """For a=b (circular), DLR = a regardless of gamma."""
        r = 4.0
        gammas = [0, np.pi / 4, np.pi / 2, np.pi]
        df = pd.DataFrame({'a': r, 'b': r, 'gamma': gammas})
        result = compute_dlr(df)
        assert np.allclose(result['dlr'].values, r, rtol=1e-10)


# ---------------------------------------------------------------------------
# find_host tests
# ---------------------------------------------------------------------------

@pytest.fixture
def two_galaxy_catalog():
    """Two galaxies; the first is much closer to the transient at (10.0, 0.0)."""
    return pd.DataFrame({
        'ra':    [10.0,  10.5],
        'dec':   [0.001, 0.0],
        'a':     [5.0,   5.0],
        'b':     [3.0,   3.0],
        'theta': [45.0,  45.0],
    })


class TestFindHost:

    def test_returns_two_dataframes(self, two_galaxy_catalog):
        result = find_host(10.0, 0.0, two_galaxy_catalog)
        assert len(result) == 2

    def test_all_hosts_has_required_columns(self, two_galaxy_catalog):
        all_hosts, _ = find_host(10.0, 0.0, two_galaxy_catalog)
        for col in ('gamma', 'dlr', 'sep', 'ddlr'):
            assert col in all_hosts.columns

    def test_all_hosts_sorted_by_ddlr(self, two_galaxy_catalog):
        all_hosts, _ = find_host(10.0, 0.0, two_galaxy_catalog)
        assert list(all_hosts['ddlr']) == sorted(all_hosts['ddlr'])

    def test_potential_hosts_below_threshold(self, two_galaxy_catalog):
        _, potential = find_host(10.0, 0.0, two_galaxy_catalog, ddlr_threshold=4.0)
        assert (potential['ddlr'] < 4.0).all()

    def test_no_potential_hosts_when_threshold_zero(self, two_galaxy_catalog):
        _, potential = find_host(10.0, 0.0, two_galaxy_catalog, ddlr_threshold=0.0)
        assert len(potential) == 0

    def test_all_hosts_length_matches_catalog(self, two_galaxy_catalog):
        all_hosts, _ = find_host(10.0, 0.0, two_galaxy_catalog)
        assert len(all_hosts) == len(two_galaxy_catalog)

    def test_does_not_mutate_input(self, two_galaxy_catalog):
        original_cols = set(two_galaxy_catalog.columns)
        _ = find_host(10.0, 0.0, two_galaxy_catalog)
        assert set(two_galaxy_catalog.columns) == original_cols

    def test_sep_is_positive(self, two_galaxy_catalog):
        all_hosts, _ = find_host(10.0, 0.0, two_galaxy_catalog)
        assert (all_hosts['sep'] >= 0).all()

    def test_ddlr_equals_sep_over_dlr(self, two_galaxy_catalog):
        all_hosts, _ = find_host(10.0, 0.0, two_galaxy_catalog)
        expected = all_hosts['sep'] / all_hosts['dlr']
        assert np.allclose(all_hosts['ddlr'].values, expected.values, rtol=1e-10)

    def test_custom_threshold(self):
        cat = pd.DataFrame({
            'ra':    [10.0, 10.0, 10.0],
            'dec':   [0.001, 0.01, 0.1],
            'a':     [5.0, 5.0, 5.0],
            'b':     [3.0, 3.0, 3.0],
            'theta': [45.0, 45.0, 45.0],
        })
        _, potential_tight = find_host(10.0, 0.0, cat, ddlr_threshold=1.0)
        _, potential_loose = find_host(10.0, 0.0, cat, ddlr_threshold=100.0)
        assert len(potential_tight) <= len(potential_loose)
