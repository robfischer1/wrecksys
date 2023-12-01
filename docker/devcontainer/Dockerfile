ARG use=gpu
FROM tensorflow/tensorflow:2.14.0-gpu-jupyter as gpu
FROM tensorflow/tensorflow:2.14.0-jupyter as cpu
FROM $use as base
RUN pip install pandas==2.1.3 pyarrow==14.0.1 "seaborn>=0.13.0" matplotlib fsspec tqdm aiohttp
COPY ./docker/devcontainer/jupyter_notebook_config.json /root/.jupyter/
COPY ./build /export
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]