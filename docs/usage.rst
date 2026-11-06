=====
Usage
=====

To use martini in a project::

    import martini

Docker Container
-----------------

Build and run martini in a container, that is defined by the Dockerfile in this repo, by running the following commands in the root of the project::

    docker build -t stirred-not-shaken .
    docker run stirred-not-shaken martini -h

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
