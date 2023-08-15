FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.12-cuda11.6.1

# set non interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies to build python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev \
    pkg-config \
    python3-dev

# copy the requirements.txt file first to avoid re-installing dependencies
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt
