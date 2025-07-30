FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

# パッケージインストール時の対話型プロンプトを無効化
ENV DEBIAN_FRONTEND=noninteractive

# 基本的な依存関係をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1-mesa-dev \
    libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# pythonコマンドでpython3が使えるようにシンボリックリンクを作成
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 作業ディレクトリを設定
WORKDIR /work

# requirements.txtを先にコピーして依存関係をインストール（ビルドキャッシュ最適化のため）
COPY docker/yolo/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
