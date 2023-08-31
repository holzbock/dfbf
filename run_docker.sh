# build the docker from the Dockerfile
# docker build -t pruning:latest .

# run the docker
docker run --gpus device=1 --device /dev/nvidia1:/dev/nvidia1 --device /dev/nvidia-modeset:/dev/nvidia-modeset --device /dev/nvidia-uvm:/dev/nvidia-uvm --device /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools --device /dev/nvidiactl:/dev/nvinvidiactl --ipc=host -it -v ./:/workspace/ pruning:latest
