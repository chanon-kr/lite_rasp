git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so

pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.6.0-cp37-cp37m-manylinux2010_x86_64.whl

pip install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl
