# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND="noninteractive"
ENV LIBGL_ALWAYS_INDIRECT=1
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}

ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update --fix-missing && \
    apt-get install -y \
                    python3-dev \
                    python3-pip \
                    git \
                    build-essential \
                    libgl1-mesa-dev \
                    mesa-utils \
                    libglu1-mesa-dev \
                    fontconfig \
                    libfreetype6-dev

RUN pip3 install PyOpenGL \
                 PyOpenGL_accelerate

# ────────── Python packages that don't cause trouble ──────────
RUN python3 -m pip install --no-cache-dir --upgrade "pip<24" && \
    python3 -m pip install --no-cache-dir PyOpenGL PyOpenGL_accelerate

RUN mkdir /f1tenth_gym

# Copy requirements first for better Docker layer caching
COPY requirements.txt /f1tenth_gym/requirements.txt

# Install Python packages from requirements
RUN cd /f1tenth_gym && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /f1tenth_gym

# Install f1tenth_gym in development mode
RUN cd /f1tenth_gym && \
    pip3 install -e .

WORKDIR /f1tenth_gym

CMD ["/bin/bash"]
