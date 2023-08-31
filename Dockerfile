ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

ARG DEBIAN_FRONTEND=noninteractive

# Fix from: https://github.com/open-mmlab/mmdetection/issues/7951#issuecomment-1122675960
# To fix GPG key error when running apt-get update 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install additionally python packages
RUN pip install --upgrade pip
RUN pip install openpifpaf==0.13.1
RUN pip install opencv-python
RUN pip install gsutil
RUN pip install pdbpp
RUN pip install imgaug
RUN pip install tensorboard
RUN pip install scikit-learn
RUN pip install tensorboardX==2.2
RUN pip install thop
RUN pip install lmdb==1.1.1
RUN pip install pickle5
RUN pip install pycocotools==2.0.2
RUN pip install tqdm==4.48.0

# Install MMCV
RUN pip install --no-cache-dir --upgrade pip wheel setuptools
RUN pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

# Install MMDetection
RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
ADD ./mmdetection/ /mmdetection/
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -r requirements/build.txt
RUN pip install --no-cache-dir -e .

