FROM datajoint/datajoint

MAINTAINER Fabian Sinz <sinz@bcm.edu>

# install tools to compile
RUN \
  apt-get update && \
  apt-get install -y -q \
    build-essential && \
  apt-get update && \
  apt-get install -y git


# install HDF5 reader and rabbit-mq client lib
RUN pip install h5py jupyter


RUN \
  pip install git+https://github.com/circstat/pycircstat.git && \
  pip install matplotlib_venn && \
  pip install tqdm

WORKDIR /src

RUN git clone https://github.com/fabiansinz/pyrelacs.git && \
    pip install -e pyrelacs

WORKDIR /efish

# Install code
COPY . /efish

RUN \
  rm -rf figures/* __pycache__ scripts/__pycache__

RUN \
  pip install -e .


# Hack to deal with weird bug that prevents running `jupyter notebook` directly
# from Docker ENTRYPOINT or CMD.
# Use dumb shell script that runs `jupyter notebook` :(
# https://github.com/ipython/ipython/issues/7062
RUN mkdir -p /scripts
ADD ./config/run_jupyter.sh /scripts/

# Add Jupyter Notebook config
ADD ./config/jupyter_notebook_config.py /root/.jupyter/

# By default start running jupyter notebook
ENTRYPOINT ["/scripts/run_jupyter.sh"]