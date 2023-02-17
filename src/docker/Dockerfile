# create virtual ubuntu with OpenCL in docker image
# FROM nvidia/opencl:latest
FROM ubuntu:20.04

# maintainer of docker file
# MAINTAINER André Pedersen <andre.pedersen@sintef.no>

# set language, format, and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# install pip and wget
RUN apt-get update -y && \
    apt-get install -y \
        python3-pip \
        wget

# Install OpenCL
RUN apt-get install intel-opencl-icd -y

# Install OpenGL
RUN apt-get install -y \
    libopengl0 \
    libgl1 \
    libusb-1.0-0-dev \
    libcairo2

# install python3 dependencies
RUN pip3 install --default-timeout=1000 pyfast==4.6.0

# CMD
# ENTRYPOINT ["runPipeline", "--datahub", "breast-tumour-segmentation", "--file"]
# CMD ["runPipeline", "--datahub", "breast-tumour-segmentation", "--file"]

# runPipeline --datahub breast-tumour-segmentation --file path/to/your/whole-slide-image.vsi
