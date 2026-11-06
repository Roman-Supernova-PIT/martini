=====
Usage
=====

To use martini in a project::

    import martini

Docker Container
-----------------

Build and run martini in a container, that is defined by the Dockerfile in this repo, by running the following commands in the root of the project::

    docker build -t stirred-not-shaken .
    docker run stirred-not-shaken --help

The image runs ``martini-run`` by default.  To run the single-position ``martini`` command instead, use
``docker run --entrypoint martini stirred-not-shaken -h``.

Host association
----------------

``compute_ddlr`` computes the directional light radius (DLR) and the normalized
separation d_DLR for every candidate galaxy, and returns the catalog sorted so
the first row is the preferred host::

    import pandas as pd
    from martini import compute_ddlr

    cat = pd.DataFrame({
        'ra':    [10.0005, 10.01],
        'dec':   [0.0002,  0.0],
        'a':     [5.0,     8.0],    # pixels
        'b':     [3.0,     4.0],    # pixels
        'theta': [45.0,    120.0],  # degrees
    })

    ranked = compute_ddlr(10.0, 0.0, cat, method='apa', pixel_scale=0.11)
    host = ranked.iloc[0]

Pass ``ddlr_threshold`` to drop candidates with ``ddlr`` at or above a cut
(4.0 follows Sullivan et al. 2006)::

    potential_hosts = compute_ddlr(10.0, 0.0, cat, ddlr_threshold=4.0)

Use ``method='ge'`` to work from SExtractor-style ellipse coefficients
(``cxx``, ``cyy``, ``cxy``) instead of ``b`` and ``theta``; ``ge_axis_ratio``
selects the ``'ghost'`` or ``'exact'`` axis-ratio convention.

Batch processing with martini-run
---------------------------------

``martini-run`` finds hosts for a whole catalog of transients.  It reads and writes parquet files, either local or on
S3 (install the ``aws`` extra for ``s3://`` paths)::

    martini-run --transients sn.parquet --galaxies gal.parquet --out out.parquet --pixel-scale 0.11

``--pixel-scale`` is required, in arcseconds per pixel.  Confirm the value for your catalog, because it changes which
hosts are found.

**Input**

.. list-table::
   :header-rows: 1

   * - Catalog
     - Required columns
   * - Transients
     - ``ra``, ``dec`` in degrees
   * - Galaxies, ``--method apa``
     - ``ra``, ``dec`` in degrees; ``a``, ``b`` in pixels; ``theta`` in degrees, image frame
   * - Galaxies, ``--method ge``
     - ``ra``, ``dec`` in degrees; ``a`` in pixels; ``cxx``, ``cyy``, ``cxy``

**Options**

.. list-table::
   :header-rows: 1

   * - Flag
     - Default
     - Meaning
   * - ``--method``
     - ``apa``
     - ``apa`` uses ``b`` and ``theta``; ``ge`` uses the ellipse coefficients
   * - ``--ge-axis-ratio``
     - ``ghost``
     - ``ghost`` or ``exact``, used only by ``--method ge``
   * - ``--ddlr``
     - 4.0
     - Maximum d_DLR for a potential host
   * - ``--radius``
     - 60
     - Cone-search radius in arcseconds; galaxies outside it are never considered
   * - ``--top-n``
     - 5
     - Maximum number of hosts kept per transient
   * - ``--n-shards``
     - 1
     - Total number of shards; required for array jobs
   * - ``--shard-index``
     - ``$AWS_BATCH_JOB_ARRAY_INDEX``, then 0
     - Shard to process

With more than one shard, each shard writes ``out/part-NNNNN-of-MMMMM.parquet``, and the ``out`` directory can be read
back as one dataset.

**Output**

Each row is one host, so a transient can have up to ``--top-n`` rows.  A transient with no potential host keeps one
row with null galaxy columns, so every input appears in the output.  All shards write the same columns:

.. list-table::
   :header-rows: 1

   * - Column
     - Meaning
   * - ``transient_index``
     - Row label from the input transient file
   * - ``transient_*``
     - Every transient column, prefixed, such as ``transient_ra``
   * - ``ddlr_rank``
     - 1 for the best host; null if the transient has no host
   * - Galaxy columns
     - Everything in the galaxy catalog; integer, boolean and text columns become nullable
   * - ``sep``
     - Separation between transient and galaxy in arcseconds
   * - ``dlr``
     - Directional light radius in arcseconds
   * - ``ddlr``
     - ``sep`` divided by ``dlr``; lower is a better match
