FROM golang:1.21-bookworm

WORKDIR /home/src/go

RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

CMD ["go", "run", "."]
