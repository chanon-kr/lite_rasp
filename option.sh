sudo apt-get update
sudo apt-get dist-upgrade

sudo pip3 install virtualenv
python3 -m venv tflite1-env
source tflite1-env/bin/activate

sudo apt-get -y install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get -y install libxvidcore-dev libx264-dev
sudo apt-get -y install qt4-dev-tools libatlas-base-dev

# Need to get an older version of OpenCV because version 4 has errors
pip3 install opencv-python==3.4.6.27
pip install pandas
pip install --index-url https://google-coral.github.io/py-repo/ tflite_runtime


python OD.py --modeldir "model\kamo01"
python OD.py --modeldir "model\kamo01" --video "testvideo\trim.mp4"