# # Running
# docker build -t stirred-not-shaken .
# docker run stirred-not-shaken martini -h

FROM python:3.14-slim-trixie

# metadata
LABEL org.opencontainers.image.authors="Benjamin Rose <Ben_Rose@baylor.edu>"
# LABEL org.opencontainers.image.description="My default science/astro container using Python 3.14."

WORKDIR /home

RUN apt update
RUN apt install -y vim

COPY . .

RUN pip install --upgrade pip

RUN pip install asdf \
astropy \
numpy \
pandas \
# "required" dependancies that are likely not required
towncrier \
setuptools_scm \
cruft \
coverage
# galsim

RUN pip install /home 

# SHELL ["/bin/bash", "--login", "-c"]
# ENTRYPOINT /bin/bash
