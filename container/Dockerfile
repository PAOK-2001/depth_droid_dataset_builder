FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu22.04

RUN apt update -y && apt upgrade -y
RUN apt install -y fish python3-pip python3-opencv gsutil
RUN apt install -y python3-tk 

WORKDIR /root/droid-example/_droid

# Temporarily mount repo and do pip install to install the model
COPY . .
RUN pip install -r requirements.txt
# Install Python dependencies
RUN pip install -e .
