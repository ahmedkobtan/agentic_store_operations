FROM docker.ci.artifacts.walmart.com/hub-docker-release-remote/library/python:3.10.14-slim-bullseye

ENV HTTP_PROXY=http://sysproxy.wal-mart.com:8080 \
    HTTPS_PROXY=http://sysproxy.wal-mart.com:8080 \
    http_proxy=http://sysproxy.wal-mart.com:8080 \
    https_proxy=http://sysproxy.wal-mart.com:8080 \
    NO_PROXY=".walmart.com, .wal-mart.com, .walmart.net, .azurewebsites.net, .azure.com, .azmk8s.io, .azure.com, .visualstudio.com, .azureml.net, localhost, 127.0.0.1" \
    no_proxy=".walmart.com, .wal-mart.com, .walmart.net, .azurewebsites.net, .azure.com, .azmk8s.io, .azure.com, .visualstudio.com, .azureml.net, localhost, 127.0.0.1"

ENV PYTHONDONTWRITEBYTECODE=1\
    PYTHONUNBUFFERED=1\
    POETRY_VERSION=1.8.4


WORKDIR /project

# Install Poetry
RUN pip install --upgrade pip && pip install poetry==${POETRY_VERSION}

RUN poetry config virtualenvs.create false

# Copy Project toml
COPY poetry.lock pyproject.toml ./

RUN poetry install --no-root --verbose


# Install dependencies
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk git wget && \
    apt-get clean

ENTRYPOINT ["bash"]
