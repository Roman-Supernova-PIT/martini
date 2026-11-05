import numpy as np
import pandas as pd
import pytest

from ..run import associate, main, shard_output_path

ARCSEC = 1 / 3600


def make_galaxies(ra, dec):
    n = len(ra)
    return pd.DataFrame({'ra': ra, 'dec': dec, 'a': [2.0] * n, 'b': [2.0] * n, 'theta': [0.0] * n})


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


class TestAssociate:

    @pytest.mark.parametrize('method', ['prost', 'dlr'])
    def test_ranks_nearest_host_first(self, method):
        galaxies = make_galaxies([10.0, 10.0], [3 * ARCSEC, ARCSEC])
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]})
        result = associate(transients, galaxies, method=method)
        assert list(result['ddlr_rank']) == [1, 2]
        assert np.isclose(result['dec'].iloc[0], ARCSEC)

    def test_top_n_limits_rows(self):
        galaxies = make_galaxies([10.0] * 3, [ARCSEC, 2 * ARCSEC, 3 * ARCSEC])
        result = associate(pd.DataFrame({'ra': [10.0], 'dec': [0.0]}), galaxies, top_n=2)
        assert len(result) == 2

    def test_cone_excludes_distant_galaxy(self):
        galaxies = pd.DataFrame({'ra': [10.0], 'dec': [10 * ARCSEC], 'a': [50.0], 'b': [50.0], 'theta': [0.0]})
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]})
        assert len(associate(transients, galaxies, radius=20.0).dropna(subset='ddlr_rank')) == 1
        assert len(associate(transients, galaxies, radius=5.0).dropna(subset='ddlr_rank')) == 0

    def test_cone_search_crosses_ra_wrap(self):
        galaxies = make_galaxies([0.0 + ARCSEC], [0.0])
        result = associate(pd.DataFrame({'ra': [360.0 - ARCSEC], 'dec': [0.0]}), galaxies)
        assert result['ddlr_rank'].iloc[0] == 1

    def test_cone_radius_is_on_sky_at_high_dec(self):
        """5 degrees of RA at dec 89.9 is about 31 arcsec on the sky."""
        galaxies = make_galaxies([15.0], [89.9])
        transients = pd.DataFrame({'ra': [10.0], 'dec': [89.9]})
        assert associate(transients, galaxies, radius=40.0, ddlr_threshold=100.0)['ddlr_rank'].iloc[0] == 1
        assert associate(transients, galaxies, radius=20.0, ddlr_threshold=100.0)['ddlr_rank'].isna().all()

    def test_hostless_transient_is_kept_with_null_rank(self):
        galaxies = make_galaxies([10.0], [ARCSEC])
        transients = pd.DataFrame({'id': ['near', 'far'], 'ra': [10.0, 200.0], 'dec': [0.0, 0.0]})
        result = associate(transients, galaxies).sort_values('transient_index')
        assert list(result['transient_id']) == ['near', 'far']
        assert result['ddlr_rank'].iloc[0] == 1
        assert pd.isna(result['ddlr_rank'].iloc[1])

    def test_keeps_input_index_as_transient_index(self):
        galaxies = make_galaxies([10.0], [ARCSEC])
        transients = pd.DataFrame({'ra': [10.0], 'dec': [0.0]}, index=[7])
        assert associate(transients, galaxies)['transient_index'].iloc[0] == 7

    def test_empty_transients(self):
        result = associate(pd.DataFrame({'ra': [], 'dec': []}), make_galaxies([10.0], [0.0]))
        assert len(result) == 0
        assert 'ddlr_rank' in result.columns


class TestShardOutputPath:

    def test_single_shard_is_unchanged(self):
        assert shard_output_path('s3://b/out.parquet', 0, 1) == 's3://b/out.parquet'

    def test_multiple_shards_write_part_files(self):
        assert shard_output_path('s3://b/out.parquet', 3, 500) == 's3://b/out/part-00003-of-00500.parquet'


class TestMain:

    def test_single_run_writes_every_transient(self, catalog_paths, tmp_path):
        transients, galaxies = catalog_paths
        out = tmp_path / 'new_dir' / 'out.parquet'
        main(['--transients', transients, '--galaxies', galaxies, '--out', str(out)])
        result = pd.read_parquet(out)
        assert sorted(result['transient_id']) == ['a', 'b', 'c', 'd']
        assert (result['ddlr_rank'] == 1).all()

    def test_shards_cover_each_transient_exactly_once(self, catalog_paths, tmp_path):
        transients, galaxies = catalog_paths
        out = tmp_path / 'out.parquet'
        for index in range(3):
            main(['--transients', transients, '--galaxies', galaxies, '--out', str(out),
                  '--shard-index', str(index), '--n-shards', '3'])
        result = pd.read_parquet(tmp_path / 'out')
        assert sorted(result['transient_id']) == ['a', 'b', 'c', 'd']

    def test_shard_index_defaults_to_batch_env(self, catalog_paths, tmp_path, monkeypatch):
        transients, galaxies = catalog_paths
        monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '1')
        out = tmp_path / 'out.parquet'
        main(['--transients', transients, '--galaxies', galaxies, '--out', str(out), '--n-shards', '2'])
        result = pd.read_parquet(tmp_path / 'out' / 'part-00001-of-00002.parquet')
        assert sorted(result['transient_id']) == ['b', 'd']

    def test_array_job_without_n_shards_is_an_error(self, catalog_paths, tmp_path, monkeypatch):
        """Otherwise each array child would silently fail or overwrite the same output."""
        transients, galaxies = catalog_paths
        monkeypatch.setenv('AWS_BATCH_JOB_ARRAY_INDEX', '4')
        with pytest.raises(SystemExit):
            main(['--transients', transients, '--galaxies', galaxies, '--out', str(tmp_path / 'out.parquet')])
