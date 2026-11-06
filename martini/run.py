"""Batch driver: associate a catalog of transients with hosts from a galaxy catalog."""

__all__ = ['main', 'associate', 'shard_output_path']

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .host_association import compute_ddlr


def _unit_vectors(ra, dec):
    """Convert RA and Dec in degrees to an ``(n, 3)`` array of unit vectors."""
    ra, dec = np.deg2rad(ra), np.deg2rad(dec)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _nullable_dtype(dtype):
    """Return a dtype that can hold nulls, for galaxy columns on hostless rows."""
    if pd.api.types.is_bool_dtype(dtype):
        return 'boolean'
    if pd.api.types.is_integer_dtype(dtype):
        return 'Int64'
    if pd.api.types.is_object_dtype(dtype):
        return 'string'
    return dtype


def associate(transients, galaxies, pixel_scale, method='apa', ge_axis_ratio='ghost', ddlr_threshold=4.0,
              radius=60.0, top_n=5):
    """Find the top host candidates for each transient.

    Parameters
    ----------
    transients : `~pandas.DataFrame`
        Must contain ``ra`` and ``dec`` in degrees.
    galaxies : `~pandas.DataFrame`
        Must contain ``ra`` and ``dec`` in degrees and ``a`` in pixels.  ``method='apa'``
        also needs ``b`` in pixels and ``theta`` in degrees (image frame), and
        ``method='ge'`` needs ``cxx``, ``cyy``, and ``cxy``.
    pixel_scale : float
        Arcseconds per pixel.
    method : {'apa', 'ge'}
        Host association method, passed to `~martini.host_association.compute_ddlr`.
    ge_axis_ratio : {'ghost', 'exact'}
        Axis-ratio convention for ``method='ge'``.
    ddlr_threshold : float
        Maximum delta-DLR for a galaxy to be a potential host.
    radius : float
        Cone-search radius in arcseconds.  Galaxies outside it are never considered, so
        it must be large enough to reach ``ddlr_threshold`` for the largest hosts.
    top_n : int
        Maximum number of hosts to keep per transient.

    Returns
    -------
    result : `~pandas.DataFrame`
        One row per host.  Transient columns are prefixed with ``transient_``, and
        ``transient_index`` is the index label of the row in ``transients``.
        ``ddlr_rank`` starts at 1.  A transient with no potential host gets one row with
        null galaxy columns and a null ``ddlr_rank``, so that every input is accounted for.
        Columns and dtypes don't depend on which hosts were found, so parts from different
        shards can be read back as one dataset.
    """
    galaxies = galaxies.reset_index(drop=True)
    tree = cKDTree(_unit_vectors(galaxies['ra'].values, galaxies['dec'].values))
    # Query by chord length, the straight-line distance between points on the unit sphere.
    chord = 2 * np.sin(np.deg2rad(radius / 3600) / 2)
    neighbors = tree.query_ball_point(_unit_vectors(transients['ra'].values, transients['dec'].values), chord)

    frames = []
    prefixed = transients.add_prefix('transient_')
    for i, (index, row) in enumerate(prefixed.iterrows()):
        found = pd.DataFrame()
        if neighbors[i]:
            potential = compute_ddlr(row['transient_ra'], row['transient_dec'], galaxies.iloc[neighbors[i]],
                                     method=method, pixel_scale=pixel_scale, ge_axis_ratio=ge_axis_ratio,
                                     ddlr_threshold=ddlr_threshold)
            found = potential.head(top_n).copy()
        if found.empty:
            found = pd.DataFrame(index=[0])
            found['ddlr_rank'] = pd.NA
        else:
            found['ddlr_rank'] = np.arange(1, len(found) + 1)
        found['transient_index'] = index
        for column, value in row.items():
            found[column] = value
        frames.append(found)

    dtypes = {'transient_index': transients.index.dtype, **prefixed.dtypes.to_dict(), 'ddlr_rank': 'Int64',
              **{column: _nullable_dtype(dtype) for column, dtype in galaxies.dtypes.items()},
              'sep': 'float64', 'dlr': 'float64', 'ddlr': 'float64'}
    with warnings.catch_warnings():
        # Null rows for transients without a host have no galaxy columns, which is intended.
        warnings.simplefilter('ignore', FutureWarning)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result.reindex(columns=list(dtypes)).astype(dtypes)


def shard_output_path(out, shard_index, n_shards):
    """Return the output path for one shard.

    A single shard writes to ``out``.  Multiple shards each write a part file under a
    directory named after ``out``, so that array jobs don't overwrite each other and
    the directory can be read back as one parquet dataset.
    """
    if n_shards == 1:
        return out
    return f'{out.removesuffix(".parquet")}/part-{shard_index:05d}-of-{n_shards:05d}.parquet'


def _default_shard_index():
    return int(os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX', 0))


def _parser():
    parser = argparse.ArgumentParser(description='Associate transients with host galaxies in batch.')
    parser.add_argument('--transients', required=True, help='Parquet path or s3:// URL with ra and dec columns.')
    parser.add_argument('--galaxies', required=True,
                        help='Parquet path or s3:// URL.  Needs ra, dec, and a, plus b and theta for --method apa, '
                             'or cxx, cyy, and cxy for --method ge.')
    parser.add_argument('--out', required=True, help='Output parquet path or s3:// URL.')
    parser.add_argument('--shard-index', type=int, default=None,
                        help='Shard to process.  Defaults to $AWS_BATCH_JOB_ARRAY_INDEX, then 0.')
    parser.add_argument('--n-shards', type=int, default=1, help='Total number of shards.')
    parser.add_argument('--pixel-scale', type=float, required=True, help='Arcseconds per pixel of a and b.')
    parser.add_argument('--method', choices=['apa', 'ge'], default='apa', help='Host association method.')
    parser.add_argument('--ge-axis-ratio', choices=['ghost', 'exact'], default='ghost',
                        help='Axis-ratio convention for --method ge.')
    parser.add_argument('--ddlr', type=float, default=4.0, help='Maximum delta-DLR for a potential host.')
    parser.add_argument('--radius', type=float, default=60.0, help='Cone-search radius in arcseconds.')
    parser.add_argument('--top-n', type=int, default=5, help='Maximum number of hosts to keep per transient.')
    return parser


def main(args=None):
    """Entry point for the ``martini-run`` command-line tool."""
    parser = _parser()
    res = parser.parse_args(args)

    shard_index = _default_shard_index() if res.shard_index is None else res.shard_index
    if res.n_shards < 1 or not 0 <= shard_index < res.n_shards:
        parser.error(f'shard index {shard_index} is not valid for {res.n_shards} shard(s); '
                     'array jobs must pass --n-shards')

    transients = pd.read_parquet(res.transients).iloc[shard_index::res.n_shards]
    galaxies = pd.read_parquet(res.galaxies)

    result = associate(transients, galaxies, res.pixel_scale, method=res.method, ge_axis_ratio=res.ge_axis_ratio,
                       ddlr_threshold=res.ddlr, radius=res.radius, top_n=res.top_n)

    out = shard_output_path(res.out, shard_index, res.n_shards)
    if '://' not in out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    result.to_parquet(out, index=False)
    print(f'shard {shard_index}/{res.n_shards}: {len(transients)} transients, {len(result)} rows -> {out}')
