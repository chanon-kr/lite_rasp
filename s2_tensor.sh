git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so
