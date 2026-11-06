# # Running
# docker build --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 -t martini .
# docker run martini --help

FROM python:3.12-slim

# metadata
LABEL org.opencontainers.image.authors="Benjamin Rose <Ben_Rose@baylor.edu>"

# .dockerignore excludes .git, so setuptools_scm can't read the version from it.
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0

WORKDIR /home

COPY . .

RUN pip install --no-cache-dir ".[aws]"

ENTRYPOINT ["martini-run"]
