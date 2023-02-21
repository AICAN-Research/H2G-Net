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

## Installing pyFAST with Docker
1. First clone github:
```
git clone
```

2. Build docker image
```
docker build -t h2gnet .
```

3. Test running test FPL with FAST in the docker image
```
docker run --rm -it --gpus all -v /tmp/.X11-unix:/tmp/.X11-unix h2gnet python3 /opt/tmp/deploy.py
```
