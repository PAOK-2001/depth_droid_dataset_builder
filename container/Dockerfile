FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu22.04

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=dev

RUN if ! getent group ${GROUP_ID} >/dev/null; then \
      groupadd -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /usr/bin/fish ${USERNAME}

# Optional: pre-create XDG dir
RUN mkdir -p /home/${USERNAME}/.xdg && \
    chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME}

RUN apt update -y && apt upgrade -y
RUN apt install -y fish python3-pip python3-opencv gsutil
RUN apt install -y python3-tk 

WORKDIR /root/droid-example/_droid

# Temporarily mount repo and do pip install to install the model
COPY . .
RUN pip install -r requirements.txt
# Install Python dependencies
RUN pip install -e .