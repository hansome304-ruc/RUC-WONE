# =============================================================================
# ZeroGrasp + perception server image (single container, RTX 3090 / sm_86)
#
# Layout overview (top -> bottom mostly stable, edits to bottom layers cheap):
#   1. system packages
#   2. GPU arch + numpy<2 anchor pin
#   3. pytorch ecosystem on cu121 wheels
#   4. project requirements.txt under -c constraints.txt
#   5. octree_feature_extractor submodule (compiled CUDA extension)
#   6. remaining grasp deps
#   7. server-side deps (FastAPI / transformers)
#   8. final numpy<2 guard (transitive deps love to drag it back)
# =============================================================================
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# --- system packages -----------------------------------------------------------
RUN apt-get update && apt-get install -y software-properties-common
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        tar xz-utils cmake curl docker.io ffmpeg git htop zsh tmux vim && \
    apt-get autoclean -y && apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ADD https://github.com/peak/s5cmd/releases/download/v2.2.2/s5cmd_2.2.2_Linux-64bit.tar.gz /opt/
RUN tar -xf /opt/s5cmd_2.2.2_Linux-64bit.tar.gz -C /opt/ && \
    mv /opt/s5cmd /usr/bin/ && rm /opt/s5cmd_2.2.2_Linux-64bit.tar.gz

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" && \
    sed -i 's|^ZSH_THEME=.*|ZSH_THEME="agnoster"|' /root/.zshrc

# --- GPU arch + numpy anchor ---------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"
RUN pip install --no-cache-dir ninja "numpy<2"

# --- pytorch ecosystem (must use cu121 wheels) --------------------------------
# The base image already bundles torch 2.2.0 cu121, but re-installing with
# explicit pins protects us from any base-image drift in the future.
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir torch-scatter \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html && \
    pip install --no-cache-dir -U xformers==0.0.24 \
        --index-url https://download.pytorch.org/whl/cu121

# --- project requirements with hard version constraints -----------------------
COPY requirements.txt constraints.txt /opt/app/
WORKDIR /opt/app
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt

# --- ocnn override ------------------------------------------------------------
# requirements.txt's `ocnn @ git+...` resolves to a 2.3+ commit that calls
# triton.Autotuner(..., pre_hook=...) — a kwarg only added in triton 3.0.
# Force-replace with the last good PyPI version that works on triton 2.2.
RUN pip install --no-cache-dir --force-reinstall --no-deps "ocnn==2.2.4"

# --- octree_feature_extractor (CUDA extension, compiled here) -----------------
# IMPORTANT: the host-side submodule MUST be checked out at `origin/main`
# (commit 5e84aea or later) so its forward signature matches zerograsp/nets/
# model.py:313. The .gitmodules pin (d602008 "Quickfix") has the OLD 8-arg
# forward and will break inference at runtime.
COPY submodules/octree_feature_extractor/ /opt/app/submodules/octree_feature_extractor
WORKDIR /opt/app/submodules/octree_feature_extractor
RUN pip install --no-cache-dir -c /opt/app/constraints.txt --no-deps -v .

# --- remaining grasp deps (--no-deps to avoid dragging numpy / torch) ---------
RUN pip install --no-cache-dir -c /opt/app/constraints.txt graspnetAPI --no-deps && \
    pip install --no-cache-dir -c /opt/app/constraints.txt \
        transforms3d autolab_core cvxopt grasp_nms && \
    pip install --no-cache-dir -c /opt/app/constraints.txt torch_cluster \
        -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# --- server-side deps (FastAPI, transformers for SAM/GD) ----------------------
# Reset WORKDIR after the ofe submodule layer (which left us in /opt/app/submodules/...).
WORKDIR /opt/app
COPY requirements-server.txt /opt/app/
RUN pip install --no-cache-dir -c /opt/app/constraints.txt -r /opt/app/requirements-server.txt

# --- final numpy<2 guard ------------------------------------------------------
# graspnetAPI / autolab_core / accelerate / transformers can all transitively
# pull numpy 2.x. Slam it back to <2 once at the very end.
RUN pip install --no-cache-dir --force-reinstall --no-deps "numpy<2"

# --- finishing touches --------------------------------------------------------
WORKDIR /opt/app
RUN git config --global --add safe.directory /opt/app && \
    git config --global --add safe.directory /opt/app/submodules/octree_feature_extractor

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=9100

EXPOSE 9100
