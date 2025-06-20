FROM stereolabs/zed:4.1-devel-cuda12.1-ubuntu22.04

ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=dev

# 1. Create dev user, give passwordless sudo, and make sure they own their home
RUN if ! getent group ${GROUP_ID} >/dev/null; then \
      groupadd -g ${GROUP_ID} ${USERNAME}; \
    fi && \
    useradd -m -u ${USER_ID} -g ${GROUP_ID} -s /usr/bin/fish ${USERNAME} && \
    apt update && apt install -y sudo && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/${USERNAME}/.xdg && \
    chown -R ${USER_ID}:${GROUP_ID} /home/${USERNAME}

# 2. Make sure dev has full access to /usr/local where system Python packages live
RUN chown -R ${USER_ID}:${GROUP_ID} /usr/local

# 3. Install base packages and Python dependencies globally
RUN apt update -y && apt upgrade -y && \
    apt install -y \
        fish python3-pip python3-opencv gsutil python3-tk \
        build-essential git curl

# 4. Copy and install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r /tmp/requirements.txt

# 5. Copy your code and install in editable mode globally
COPY . /opt/droid
WORKDIR /opt/droid
RUN pip3 install -e .

# 6. Switch to dev user
USER ${USERNAME}
WORKDIR /home/${USERNAME}/droid-example/_droid

# 7. Runtime environment
ENV HOME=/home/${USERNAME}
ENV XDG_RUNTIME_DIR=$HOME/.xdg
ENV PATH="$HOME/.local/bin:$PATH"
