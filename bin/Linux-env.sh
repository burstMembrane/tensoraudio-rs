#!/bin/bash
TARGET_PATH=`pwd`/libtorch

if [ -d "$TARGET_PATH" ]; then
  echo "libtorch already exists"
  export LIBTORCH=$TARGET_PATH
  export LIBTORCH_CXX11_ABI=0
  export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
  return
fi

# Check for CUDA
if command -v nvcc &> /dev/null; then
  PYTORCH_VERSION="2.6.0"
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
if [ -z "$CUDA_VERSION" ]; then
  echo "CUDA not found, downloading CPU version"
  DOWNLOAD_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcpu.zip"
else
  CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
  CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
  COMBINED_CUDA_VERSION="$CUDA_MAJOR$CUDA_MINOR"
  echo "CUDA_VERSION: $CUDA_VERSION"
  echo "CUDA_MAJOR: $CUDA_MAJOR"
  echo "CUDA_MINOR: $CUDA_MINOR"
  DOWNLOAD_URL="https://download.pytorch.org/libtorch/cu$COMBINED_CUDA_VERSION/libtorch-shared-with-deps-${PYTORCH_VERSION}%2Bcu$COMBINED_CUDA_VERSION.zip"
fi
echo "Downloading LibTorch from: $DOWNLOAD_URL"
wget $DOWNLOAD_URL -O libtorch.zip
unzip libtorch.zip
rm libtorch.zip

export LIBTORCH=`pwd`/libtorch
export LIBTORCH_CXX11_ABI=0
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH