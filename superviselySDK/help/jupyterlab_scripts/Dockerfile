ARG REGISTRY
ARG TAG
FROM ${REGISTRY}/base-jupyterlab:${TAG}

RUN pip install --no-cache-dir \
	requests==2.19.1 \
	requests-toolbelt

RUN pip install tqdm
RUN apt-get install tree

RUN pip install python-slugify
RUN pip install seaborn

############### copy code ###############
ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/supervisely_lib

ENV PYTHONPATH /workdir:/workdir/src:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir

EXPOSE 8888
