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
