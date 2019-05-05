# TODO change this to public base.
# FROM docker.deepsystems.io/supervisely/five/base-pytorch:4.0.0
ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-pytorch-v04:${TAG}

ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/libs/supervisely_lib

ENV PYTHONPATH /workdir/src:/workdir/libs:/workdir/libs/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src