ARG use=gpu
# DL3026 is exempt in this repo (Rob, 2026-09-15): wrecksys runs on AWS with no
# access to the fleet's internal registry proxy, so its bases come from public registries.
# hadolint ignore=DL3026
FROM tensorflow/tensorflow:2.14.0-gpu-jupyter as gpu
# hadolint ignore=DL3026
FROM tensorflow/tensorflow:2.14.0-jupyter as cpu
# hadolint ignore=DL3026
FROM $use as base
RUN pip install pandas==2.1.3 pyarrow==14.0.1 "seaborn>=0.13.0" matplotlib fsspec tqdm aiohttp
COPY ./docker/devcontainer/jupyter_notebook_config.json /root/.jupyter/
COPY ./build /export
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]