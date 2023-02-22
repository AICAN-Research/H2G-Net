# build custom image from the ubuntu docker image
#FROM ubuntu:20.04
#FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
#FROM tensorflow/tensorflow:2.8.0-gpu
FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# maintainer of docker file
# MAINTAINER André Pedersen <andre.pedersen@sintef.no>

# set language, format, and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# install dependencies and OpenCL
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y \
    apt-utils \
    unzip \
    tar \
    curl \
    xz-utils \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    python3-pip \
    ;
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install OpenGL
RUN apt-get install -y \
    libopengl0 \
    libgl1 \
    libusb-1.0-0-dev \
    libcairo2

# Install convenience stuff
RUN apt-get install -y \
    wget \
    vim

# install python3 dependencies
RUN pip3 install --default-timeout=1000 pyfast==4.6.0

# Copy shit
COPY pipelines/ /opt/pipelines/
COPY applications/ /opt/applications/

# Download test sample and place it in /pipelines/
RUN cd /opt/pipelines/ && wget https://folk.ntnu.no/andpeder/FastPathology/test_data/WSI/A05.svs
