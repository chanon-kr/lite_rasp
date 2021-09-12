# Update
sudo apt-get update
sudo apt-get dist-upgrade

# V env
sudo pip3 install virtualenv
python3 -m venv tflite
source tflite/bin/activate

# Open CV
sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev
pip install opencv-python

# Build Tensorflow from Source
pip install https://github.com/bitsy-ai/tensorflow-arm-bin/releases/download/v2.4.0-rc2/tensorflow-2.4.0rc2-cp37-none-linux_armv7l.whl

# Run
python3 OD.py --modeldir "model/kamo01"