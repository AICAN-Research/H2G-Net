# Docker test

## Setup Docker to work with FAST

The recommended way of installing docker and docker compose is through docker desktop (for more details see [documentations](https://docs.docker.com/compose/install/)).

To install docker dekstop, you will need to install and setup some things on the machine itself, before installing docker desktop. For Linux, see [here](https://docs.docker.com/desktop/install/linux-install/).

Note that if you install docker desktop, `docker compose` (without "-") is included. **Don't** install it separately, as you could easily install an outdated version which might not be compatible with this current setup.

Even still, after that is done, you might have to install `nvidia-docker2` to enable docker to use the NVIDIA GPU for inference:
```
sudo apt update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Build Docker image with pyFAST

1. First clone github:
```
git clone
```

2. Build docker image
```
docker build -t h2gnet .
```

## Verifying that everything works

Test that OpenCL is properly installed:

```
docker run --rm -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix h2gnet clinfo
```

If you get 0 support OpenCL platforms, then it did not. Otherwise, check if you can import pyFAST:

```
docker run --rm -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix h2gnet python3 -c "import fast; print("It worked!")"
```

If you did not get the `It worked!` print, then import failed. Lastly, try to run the tissue segmentation pipeline:

```
docker run --rm -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix h2gnet python3 /opt/tmp/deploy.py
```

If `True` was prompted at the end, the FPL generated a result on disk, and then FAST is properly setup with Docker!
