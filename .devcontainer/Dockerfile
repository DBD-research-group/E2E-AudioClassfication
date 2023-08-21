FROM pytorchlightning/pytorch_lightning:base-cuda-py3.10-torch2.0-cuda11.7.1

# set non interactive mode
ARG DEBIAN_FRONTEND=noninteractive

# install dependencies to build python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcairo2-dev \
    pkg-config \
    python3-dev

# install python packages
RUN pip install --upgrade pip
RUN pip install ipykernel
RUN pip install torchvision
RUN pip install lightning
RUN pip install datasets
RUN pip install torchaudio
RUN pip install librosa
RUN pip install soundfile

# copy the requirements.txt file first to avoid re-installing dependencies
COPY requirements.txt .

# install dependencies
# RUN pip install -r requirements.txt
