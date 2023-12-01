FROM tensorflow/serving as base
ENV MODEL_NAME=wrecksys
COPY /build/model /models/wrecksys/001

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]