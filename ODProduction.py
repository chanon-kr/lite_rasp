#!/usr/bin/env python
# coding: utf-8

## I Modify This Script from Script from Link Below
### https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/blob/master/TFLite_detection_video.py

# Import packages
import os,json ,cv2 , sys, argparse, shutil
import numpy as np
import importlib.util
from datetime import datetime, timedelta
from py_topping.data_connection.gcp import da_tran_bucket
from glob import glob

with open('config.json', 'rb') as f :
    prep_config = json.load(f)

gcs = da_tran_bucket(project_id = prep_config["gcp_projectid"] 
                    , bucket_name = prep_config["gcp_bucket"] 
                    , credential = prep_config["gcp_credential"] )

model_folder = prep_config["model_folder"].split('model')[-1]
if not os.path.isdir('model{}'.format(model_folder)) : 
    os.mkdir('model{}'.format(model_folder))

for file in ['model.tflite','label.txt','model_config.json'] :
    gcs.download(bucket_file = prep_config["model_folder"]  + '/{}'.format(file)
                 , local_file = 'model{}/{}'.format(model_folder,file)) 

with open('model{}/model_config.json'.format(model_folder), 'rb') as f :
    config = json.load(f)

## Define and parse input arguments
MODEL_NAME = config["modeldir"]
GRAPH_NAME = config["graph"]
LABELMAP_NAME = config["labels"]
VIDEO_NAME = config["video"]
min_conf_threshold = float(config["threshold"])
x1,x2 = (float(i) for i in config["xcrop"].split(','))
y1,y2 = (float(i) for i in config["ycrop"].split(','))
use_TPU = bool(config["edgetpu"])
save_slot = int(config["save_slot"])
gcp_folder = config["gcp_folder"]
gcp_projectid = config["gcp_projectid"] 
gcp_bucket = config["gcp_bucket"]
gcp_credential = config["gcp_credential"]
save_size = float(config["save_size"])
show_window = bool(config["show_window"])
restart_limit = float(config["restart_hour"])
# In[2]:


# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'rb') as f:
    labels = json.load(f)
labels = {int(i) : j for i, j in labels.items()}

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if 0 in labels :
    if labels[0] == '???':
        del(labels[0])
labels = [i[1] for i in sorted(labels.items())]

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)
input_mean = 127.5
input_std = 127.5

# Open video file
if 'rtsp' in VIDEO_NAME :
    video = cv2.VideoCapture()
    video.open(VIDEO_NAME)
elif VIDEO_NAME != '0' :
    video = cv2.VideoCapture(VIDEO_PATH)
else :
    video = cv2.VideoCapture(0)

# Video Size
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
y1,y2,x1,x2 = int(imH*y1),int(imH*y2), int(imW*x1),int(imW*x2)

# Prepare Save Slot
now = datetime.now()
now_minute = now.minute
now_slot = (now.replace(minute = 0) + timedelta(minutes = int(now_minute/save_slot)*save_slot)).strftime('%Y%m%d%H%M')
fps_list = []

# Prepare Out Folder
for i in ['tmp','tmp/Found','tmp/All'] :
    if not os.path.isdir(i) : os.mkdir(i)

start_time = datetime.now()
while(video.isOpened()):
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    now = datetime.now()
    if not ret:
        print('Reached the end of the video!')
        break

    # Create Filename
    filename = now.strftime('%Y%m%d%H%M%S')
    # Create Check Point
    now_minute = now.minute
    temp_slot = (now.replace(minute = 0) + timedelta(minutes = int(now_minute/save_slot)*save_slot)).strftime('%Y%m%d%H%M')

    if now_slot != temp_slot : 
        # Create Connection
        print('Begin Save at',datetime.now())
        gcs = da_tran_bucket(project_id = gcp_projectid
                            , bucket_name = gcp_bucket
                            , credential = gcp_credential )

        # Update Time
        with open("tmp/lastupload.txt" , "w") as f :
            f.write(now_slot)
        gcs.upload(bucket_file = '{}/{}'.format(gcp_folder,"lastupload_begin.txt")
                    ,local_file = "tmp/lastupload.txt")
        os.remove("tmp/lastupload.txt")
        # Save as Video (Save Space)
        if len(fps_list) > 0 : 
            fps = np.mean(fps_list)
            print('Average FPS :',fps)

            img_array_i = []
            for i in glob('tmp/All/*.png'):
                img_i = cv2.imread(i)
                height_i, width_i, layers_i = img_i.shape
                size_i = (width_i,height_i)
                img_array_i.append(img_i)
            out = cv2.VideoWriter('tmp/clip.avi',cv2.VideoWriter_fourcc(*'DIVX'), fps, size_i)   
            for i in range(len(img_array_i)):
                out.write(img_array_i[i])
            out.release()
            del img_array_i
            gcs.upload(bucket_file = '{}/Clip/clip_{}.avi'.format(gcp_folder,now_slot)
                        , local_file = 'tmp/clip.avi')
            for i in glob('tmp/All/*.png') : os.remove(i)
            os.remove('tmp/clip.avi')
        # Break Time
        cal_restart = (now - start_time).total_seconds()/3600
        if cal_restart > restart_limit : break

        # Save Found Picture
        shutil.make_archive('tmp/Found', 'zip', 'tmp/Found')
        gcs.upload(bucket_file = '{}/Found/Found_{}.zip'.format(gcp_folder,now_slot)
                    , local_file = 'tmp/Found.zip')
        # os.remove('tmp/Found.zip')
        for i in glob('tmp/Found/*.png') : os.remove(i)
        print('End Save at',datetime.now())

        # Update Time
        with open("tmp/lastupload.txt" , "w") as f :
            f.write(now_slot)
        gcs.upload(bucket_file = '{}/{}'.format(gcp_folder,"lastupload_end.txt")
                    ,local_file = "tmp/lastupload.txt")
        os.remove("tmp/lastupload.txt")

    now_slot = (now.replace(minute = 0) + timedelta(minutes = int(now_minute/save_slot)*save_slot)).strftime('%Y%m%d%H%M')
    show = 0
    
    # Prep Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = frame_rgb[y1:y2,x1:x2]
    frame_resized = cv2.resize(frame_resized, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Solve Error (I think it's a bug) from some TFlite model
    if 'lite' not in MODEL_NAME :
        # Retrieve detection results : GCP AutoML
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    else :
        # For Some TFLite : efficientdet_lite0, 1, 2
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
            # print(count_frame , 'found')
            show += 1
            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * (y2-y1)))) + y1
            xmin = int(max(1,(boxes[i][1] * (x2-x1)))) + x1
            ymax = int(min((y2-y1),(boxes[i][2] * (y2-y1)))) + y1
            xmax = int(min((x2-x1),(boxes[i][3] * (x2-x1)))) + x1
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 4)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    
    # Add rectangle of Applied Area
    cv2.rectangle(frame, (x2,y2), (x1,y1), (0, 255, 255), 2)
    # All the results have been drawn on the frame, so it's time to display it.
    if show_window : cv2.imshow('Object detector', frame)
    
    # To Save Space
    frame = cv2.resize(frame, (int(imW*save_size), int(imH*save_size)))
    # Save image
    savename = 'tmp/All/at_{}.png'.format(filename) 
    cv2.imwrite(savename, frame)
    # Duplicate Picture with Label in other folder 
    if show > 0 : 
        savename = 'tmp/Found/at_{}.png'.format(filename)
        cv2.imwrite(savename, frame)

    # Save for cal fps
    fps_cal = (datetime.now() - now).total_seconds()
    if fps_cal == 0 : fps_cal = 60
    else : fps_cal = 1/fps_cal
    fps_list.append(fps_cal)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        print('break')
        break

# Clean up
video.release()
cv2.destroyAllWindows()