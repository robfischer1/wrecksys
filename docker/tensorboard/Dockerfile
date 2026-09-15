# DL3026 is exempt in this repo (Rob, 2026-09-15): wrecksys runs on AWS with no
# access to the fleet's internal registry proxy, so its bases come from public registries.
# hadolint ignore=DL3026
FROM python:3.10 as base

RUN apt-get update && apt-get -y --no-install-recommends install \
    tini \
    && rm -rf /var/lib/apt/lists /var/cache/apt/archives

RUN pip install tensorboard
EXPOSE 6006

ENTRYPOINT ["/usr/bin/tini", "--"]

CMD ["tensorboard", "serve", "--logdir=/logdir", "--bind_all"]