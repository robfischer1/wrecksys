# DL3026 is exempt in this repo (Rob, 2026-09-15): wrecksys runs on AWS with no
# access to the fleet's internal registry proxy, so its bases come from public registries.
# hadolint ignore=DL3026
FROM tensorflow/serving as base
ENV MODEL_NAME=wrecksys
COPY /build/model /models/wrecksys/001

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]