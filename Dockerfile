FROM ubuntu

COPY . /src/SMEFT19/
RUN apt update && apt install -y \
    git \
    python3.8 \
    python3-pip \
    && pip install -r /src/SMEFT19/requirements.txt \
    /src/SMEFT19