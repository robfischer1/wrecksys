FROM ubuntu:latest as base
RUN apt-get update && apt-get -y --no-install-recommends install \
    python3-venv \
    && rm -rf /var/lib/apt/lists /var/cache/apt/archives

RUN groupadd -g 1000 app && \
    useradd -u 1000 -g 1000 app

WORKDIR /home/app
RUN python3 -m venv /home/app/venv && chown -RH app:app /home/app
ENV PATH="/home/app/venv/bin:$PATH"

# COPY requirements.txt .
# RUN pip install -r requirements.txt
RUN pip install notebook

# Add Tini. Tini operates as a process subreaper for jupyter. This prevents
# kernel crashes.
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]

EXPOSE 8888
ENV PORT 8888
ENV HOSTNAME "0.0.0.0"

USER 1000
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8888","--IdentityProvider.token=''", "--ServerApp.password=''"]