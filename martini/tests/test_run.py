import numpy as np
import pandas as pd
import pytest

from ..host_association import compute_ddlr
from ..run import associate, main, shard_output_path

ARCSEC = 1 / 3600


def make_galaxies(ra, dec, a=2.0, b=2.0, theta=0.0):
    n = len(ra)
    return pd.DataFrame({'ra': ra, 'dec': dec, 'a': [a] * n, 'b': [b] * n, 'theta': [theta] * n})


def with_coefficients(galaxies):
    """Add the SExtractor ellipse coefficients that match ``a``, ``b``, and ``theta``."""
    theta = np.deg2rad(galaxies['theta'])
    a2, b2 = galaxies['a'] ** 2, galaxies['b'] ** 2
    return galaxies.assign(cxx=np.cos(theta) ** 2 / a2 + np.sin(theta) ** 2 / b2,
                           cyy=np.sin(theta) ** 2 / a2 + np.cos(theta) ** 2 / b2,
                           cxy=2 * np.cos(theta) * np.sin(theta) * (1 / a2 - 1 / b2))


@pytest.fixture
def catalog_paths(tmp_path):
    """Four transients, each 1 arcsec from its own host."""
    ra = [10.0, 20.0, 30.0, 40.0]
    transients = pd.DataFrame({'id': ['a', 'b', 'c', 'd'], 'ra': ra, 'dec': [0.0] * 4})
    galaxies = make_galaxies(ra, [ARCSEC] * 4)
    transients_path, galaxies_path = tmp_path / 'sn.parquet', tmp_path / 'gal.parquet'
    transients.to_parquet(transients_path)
    galaxies.to_parquet(galaxies_path)
    return str(transients_path), str(galaxies_path)


def run_args(catalog_paths, out):
    transients, galaxies = catalog_paths
    return ['--transients', transients, '--galaxies', galaxies, '--out', str(out), '--pixel-scale', '1.0']


class TestAssociate:

    @pytest.mark.parametrize('method', ['apa', 'ge'])
    def test_ranks_nearest_host_first(self, method):
        galaxies = with_coefficients(make_galaxies([10.0, 10.0], [3 * ARCSEC, ARCSEC], b=1.0))
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]})
        result = associate(transients, galaxies, 1.0, method=method, ge_axis_ratio='exact')
        assert list(result['ddlr_rank']) == [1, 2]
        assert np.isclose(result['dec'].iloc[0], ARCSEC)

    def test_matches_compute_ddlr(self):
        galaxies = make_galaxies([10.0, 10.0 + ARCSEC], [2 * ARCSEC, 0.0], a=3.0, b=1.0, theta=30.0)
        result = associate(pd.DataFrame({'ra': [10.0], 'dec': [0.0]}), galaxies, 0.5, ddlr_threshold=100.0)
        expected = compute_ddlr(10.0, 0.0, galaxies, pixel_scale=0.5, ddlr_threshold=100.0)
        assert np.allclose(result['ddlr'], expected['ddlr'])
        assert np.allclose(result['dlr'], expected['dlr'])

    def test_pixel_scale_sets_dlr_in_arcsec(self):
        galaxies = make_galaxies([10.0], [ARCSEC], a=10.0, b=10.0)
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]})
        coarse = associate(transients, galaxies, 1.0, ddlr_threshold=100.0)
        fine = associate(transients, galaxies, 0.1, ddlr_threshold=100.0)
        assert np.isclose(coarse['dlr'].iloc[0], 10.0)
        assert np.isclose(fine['dlr'].iloc[0], 1.0)
        assert np.isclose(fine['ddlr'].iloc[0], 10 * coarse['ddlr'].iloc[0])

    def test_top_n_limits_rows(self):
        galaxies = make_galaxies([10.0] * 3, [ARCSEC, 2 * ARCSEC, 3 * ARCSEC])
        result = associate(pd.DataFrame({'ra': [10.0], 'dec': [0.0]}), galaxies, 1.0, top_n=2)
        assert len(result) == 2

    def test_cone_excludes_distant_galaxy(self):
        galaxies = make_galaxies([10.0], [10 * ARCSEC], a=50.0, b=50.0)
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]})
        assert len(associate(transients, galaxies, 1.0, radius=20.0).dropna(subset='ddlr_rank')) == 1
        assert len(associate(transients, galaxies, 1.0, radius=5.0).dropna(subset='ddlr_rank')) == 0

    def test_cone_search_crosses_ra_wrap(self):
        galaxies = make_galaxies([0.0 + ARCSEC], [0.0])
        result = associate(pd.DataFrame({'ra': [360.0 - ARCSEC], 'dec': [0.0]}), galaxies, 1.0)
        assert result['ddlr_rank'].iloc[0] == 1

    def test_cone_radius_is_on_sky_at_high_dec(self):
        """5 degrees of RA at dec 89.9 is about 31 arcsec on the sky."""
        galaxies = make_galaxies([15.0], [89.9])
        transients = pd.DataFrame({'ra': [10.0], 'dec': [89.9]})
        assert associate(transients, galaxies, 1.0, radius=40.0, ddlr_threshold=100.0)['ddlr_rank'].iloc[0] == 1
        assert associate(transients, galaxies, 1.0, radius=20.0, ddlr_threshold=100.0)['ddlr_rank'].isna().all()

    def test_hostless_transient_is_kept_with_null_rank(self):
        galaxies = make_galaxies([10.0], [ARCSEC])
        transients = pd.DataFrame({'id': ['near', 'far'], 'ra': [10.0, 200.0], 'dec': [0.0, 0.0]})
        result = associate(transients, galaxies, 1.0).sort_values('transient_index')
        assert list(result['transient_id']) == ['near', 'far']
        assert result['ddlr_rank'].iloc[0] == 1
        assert pd.isna(result['ddlr_rank'].iloc[1])

    def test_candidates_beyond_threshold_are_kept_as_hostless(self):
        galaxies = make_galaxies([10.0], [50 * ARCSEC])
        result = associate(pd.DataFrame({'ra': [10.0], 'dec': [0.0]}), galaxies, 1.0)
        assert len(result) == 1
        assert pd.isna(result['ddlr_rank'].iloc[0])

    @pytest.mark.parametrize('ra', [[10.0], [200.0], []])
    def test_columns_and_dtypes_do_not_depend_on_hosts(self, ra):
        galaxies = make_galaxies([10.0], [ARCSEC]).assign(name=['g1'], gid=[1], good=[True])
        reference = associate(pd.DataFrame({'ra': [10.0], 'dec': [0.0]}), galaxies, 1.0)
        result = associate(pd.DataFrame({'ra': ra, 'dec': [0.0] * len(ra)}), galaxies, 1.0)
        assert result.dtypes.equals(reference.dtypes)

    def test_keeps_input_index_as_transient_index(self):
        galaxies = make_galaxies([10.0], [ARCSEC])
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]}, index=[7])
        assert associate(transients, galaxies, 1.0)['transient_index'].iloc[0] == 7

    def test_empty_transients(self):
        result = associate(pd.DataFrame({'ra': [], 'dec': []}), make_galaxies([10.0], [0.0]), 1.0)
        assert len(result) == 0
        assert 'ddlr_rank' in result.columns


class TestShardOutputPath:

    def test_single_shard_is_unchanged(self):
        assert shard_output_path('s3://b/out.parquet', 0, 1) == 's3://b/out.parquet'

    def test_multiple_shards_write_part_files(self):
        assert shard_output_path('s3://b/out.parquet', 3, 500) == 's3://b/out/part-00003-of-00500.parquet'


class TestMain:

    def test_single_run_writes_every_transient(self, catalog_paths, tmp_path):
        out = tmp_path / 'new_dir' / 'out.parquet'
        main(run_args(catalog_paths, out))
        result = pd.read_parquet(out)
        assert sorted(result['transient_id']) == ['a', 'b', 'c', 'd']
        assert (result['ddlr_rank'] == 1).all()

    def test_shards_cover_each_transient_exactly_once(self, catalog_paths, tmp_path):
        out = tmp_path / 'out.parquet'
        for index in range(3):
            main([*run_args(catalog_paths, out), '--shard-index', str(index), '--n-shards', '3'])
        result = pd.read_parquet(tmp_path / 'out')
        assert sorted(result['transient_id']) == ['a', 'b', 'c', 'd']

    def test_hostless_shard_reads_back_with_the_others(self, tmp_path):
        transients_path, galaxies_path = tmp_path / 'sn.parquet', tmp_path / 'gal.parquet'
        pd.DataFrame({'id': ['near', 'far'], 'ra': [10.0, 200.0], 'dec': [0.0, 0.0]}).to_parquet(transients_path)
        make_galaxies([10.0], [ARCSEC]).assign(name=['g1']).to_parquet(galaxies_path)
        out = tmp_path / 'out.parquet'
        for index in range(2):
            main(['--transients', str(transients_path), '--galaxies', str(galaxies_path), '--out', str(out),
                  '--pixel-scale', '1.0', '--shard-index', str(index), '--n-shards', '2'])
        result = pd.read_parquet(tmp_path / 'out').sort_values('transient_index')
        assert list(result['transient_id']) == ['near', 'far']
        assert result['ddlr'].notna().tolist() == [True, False]
        assert result['name'].iloc[0] == 'g1'

    def test_shard_index_defaults_to_batch_env(self, catalog_paths, tmp_path, monkeypatch):
        monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '1')
        main([*run_args(catalog_paths, tmp_path / 'out.parquet'), '--n-shards', '2'])
        result = pd.read_parquet(tmp_path / 'out' / 'part-00001-of-00002.parquet')
        assert sorted(result['transient_id']) == ['b', 'd']

    def test_array_job_without_n_shards_is_an_error(self, catalog_paths, tmp_path, monkeypatch):
        """Otherwise each array child would silently fail or overwrite the same output."""
        monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '4')
        with pytest.raises(SystemExit):
            main(run_args(catalog_paths, tmp_path / 'out.parquet'))

    def test_pixel_scale_is_required(self, catalog_paths, tmp_path):
        transients, galaxies = catalog_paths
        with pytest.raises(SystemExit):
            main(['--transients', transients, '--galaxies', galaxies, '--out', str(tmp_path / 'out.parquet')])

    def test_old_method_names_are_rejected(self, catalog_paths, tmp_path):
        with pytest.raises(SystemExit):
            main([*run_args(catalog_paths, tmp_path / 'out.parquet'), '--method', 'prost'])

    @pytest.mark.parametrize('convention, expected', [('exact', 2.0), ('ghost', 2 / 3)])
    def test_ge_axis_ratio_reaches_compute_ddlr(self, catalog_paths, tmp_path, convention, expected):
        """A 6x2 galaxy with its major axis north, and a transient along its minor axis."""
        transients, _ = catalog_paths
        galaxies_path = tmp_path / 'ge.parquet'
        with_coefficients(make_galaxies([10.0 + ARCSEC], [0.0], a=6.0, b=2.0, theta=90.0)).to_parquet(galaxies_path)
        out = tmp_path / 'out.parquet'
        main(['--transients', transients, '--galaxies', str(galaxies_path), '--out', str(out), '--pixel-scale', '1.0',
              '--method', 'ge', '--ge-axis-ratio', convention])
        result = pd.read_parquet(out)
        assert np.isclose(result.dropna(subset='ddlr_rank')['dlr'].iloc[0], expected)
