git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/c:libtensorflowlite_c.so

pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.6.0-cp37-cp37m-manylinux2010_x86_64.whl