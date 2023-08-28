# End-to-End Audio Transformer PyTorch Lightning implementation

This repository contains a simple PyTorch [Lightning](https://github.com/Lightning-AI/lightning) implementation of the paper [End-to-End Audio Transformer](https://arxiv.org/abs/2204.11479) by Avi Gazneli, Gadi Zimerman, Tal Ridnik, Gilad Sharir, Asaf Noy.

Original code can be found at https://github.com/Alibaba-MIIL/AudioClassfication

## Setup

See ``.devcontainer/README.md`` for instructions on how to setup the development environment.

## Data

We use the [ESC-50 dataset](https://github.com/karolpiczak/ESC-50) downloaded from 
[Huggingface](https://huggingface.co/datasets/ashraq/esc50) "ashraq/esc50" for testing.

## Notebook

See `notebooks/lightning.ipynb` for a simple example of how to train the model with Lightning.

