FROM nvidia/cuda:12.5.0-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y sudo git curl python3 python3-pip \
    iproute2 iputils-ping net-tools \
    openvswitch-switch \
    && apt-get clean

# Mininet のインストール
RUN git clone https://github.com/mininet/mininet.git && \
    cd mininet && \
    util/install.sh -a

# 必要なPythonパッケージをインストール（CUDA対応PyTorch）
RUN pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu125
RUN pip3 install numpy mininet networkx matplotlib

# スタートアップスクリプトをコピー
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]