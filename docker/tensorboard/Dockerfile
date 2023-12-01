FROM python:3.10 as base

RUN apt-get update && apt-get -y --no-install-recommends install \
    tini \
    && rm -rf /var/lib/apt/lists /var/cache/apt/archives

RUN pip install tensorboard
EXPOSE 6006

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["tensorboard", "serve", "--logdir=/logdir", "--bind_all"]