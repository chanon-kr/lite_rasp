# Before Start
## Download Raspberry PI Imager
## For 32 bit like models from TF Model Marker => Just use 32 bit
## For 64 bit like models from GCP AutoML Edge => Use 64 bit
## Raspberry Pi 4, 64 Bit Image ;
### => https://www.raspberrypi.org/forums/viewtopic.php?t=275370
### => https://downloads.raspberrypi.org/raspios_arm64/images/

# Start
sudo apt-get update
sudo apt-get dist-upgrade

# Library
pip3 install opencv-python 
pip3 install py-topping
pip3 install google-cloud-storage

# TFLite Run Time for Linux
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime