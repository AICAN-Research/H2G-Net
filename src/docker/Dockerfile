# create virtual ubuntu in docker image
# FROM ubuntu:20.04
FROM nvidia/opencl:latest

# maintainer of docker file
# MAINTAINER André Pedersen <andre.pedersen@sintef.no>

# set language, format, and stuff
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# install pip and wget
RUN apt-get update -y
RUN apt-get install python3-pip -y
RUN apt-get install wget -y

# Install OpenCL
#RUN mkdir neo
#RUN cd neo
#RUN wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-gmmlib_18.4.1_amd64.deb
#RUN wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-igc-core_18.50.1270_amd64.deb
#RUN wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-igc-opencl_18.50.1270_amd64.deb
#RUN wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-opencl_19.07.12410_amd64.deb
#RUN wget https://github.com/intel/compute-runtime/releases/download/19.07.12410/intel-ocloc_19.07.12410_amd64.deb
#RUN apt-get install ./*.deb -y

# Install OpenGL
#RUN apt-get install libopengl0 -y
#RUN apt-get install libgl1 -y
#RUN apt-get install libusb-1.0-0-dev -y
#RUN apt-get install libcairo2 -ys

# install python3 dependencies
RUN pip3 install pyfast==4.6.0

# CMD
# ENTRYPOINT ["runPipeline", "--datahub", "breast-tumour-segmentation", "--file"]
# CMD ["runPipeline", "--datahub", "breast-tumour-segmentation", "--file"]

# runPipeline --datahub breast-tumour-segmentation --file path/to/your/whole-slide-image.vsi
