FROM mcr.microsoft.com/devcontainers/base:debian

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    make \
    libx11-dev \
    gdb \
    && rm -rf /var/lib/apt/lists/*
