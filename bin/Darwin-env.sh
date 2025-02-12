#!/bin/bash
TARGET_PATH=`pwd`/libtorch


set_env_vars() {
  export LIBTORCH=$TARGET_PATH
  export LIBTORCH_CXX11_ABI=1
  export DYLD_LIBRARY_PATH=$LIBTORCH/lib:$DYLD_LIBRARY_PATH
  export LIBTORCH_BYPASS_VERSION_CHECK=1
}

if [ -d "$TARGET_PATH" ]; then
  echo "Libtorch is already downloaded and installed at $TARGET_PATH"
  set_env_vars

  return
fi

# otherwise download and unzip
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.6.0.zip
unzip libtorch-macos-arm64-2.6.0.zip
set_env_vars
# clean up
rm libtorch-macos-arm64-2.6.0.zip
