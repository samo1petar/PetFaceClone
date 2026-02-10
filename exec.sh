#!/bin/sh
docker run -it --gpus all --shm-size 64G \
    -v /home/alfred/Projects/PetFace:/workspace/ \
    pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime bash

