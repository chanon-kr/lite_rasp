#!/usr/bin/env python
# coding: utf-8

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys, json
import importlib.util


# # Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='model.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='label.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='0')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')
parser.add_argument('--xcrop', help='Crop Frame in X',
                    default='0,1')
parser.add_argument('--ycrop', help='Crop Frame in Y',
                    default='0,1')

args = parser.parse_args()
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)
x1,x2 = (float(i) for i in args.xcrop.split(','))
y1,y2 = (float(i) for i in args.ycrop.split(','))
# use_TPU = args.edgetpu

## For Local Test
# MODEL_NAME = 'model/test_pompom'
# VIDEO_NAME = '0'
# GRAPH_NAME = 'model.tflite'
# LABELMAP_NAME = 'label.txt'
# min_conf_threshold = float(0.5)
use_TPU = False

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

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'   

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to video file
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
# with open(PATH_TO_LABELS, 'r') as f:
#     labels = [line.strip() for line in f.readlines()]
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
if VIDEO_NAME != '0' :
    video = cv2.VideoCapture(VIDEO_PATH)
else :
    video = cv2.VideoCapture(0)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
y1,y2,x1,x2 = int(imH*y1),int(imH*y2), int(imW*x1),int(imW*x2)

count_frame = 0
while(video.isOpened()):
    count_frame += 1
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = video.read()
    show = 0
    if not ret:
        print('Reached the end of the video!')
        break
    # frame = frame[y1:y2,x1:x2]
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

    if 'lite' not in MODEL_NAME :
        # Retrieve detection results : GCP AutoML
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
    else :
        # For Some TFLite : efficientdet_lite2
        boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

    #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
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
    cv2.rectangle(frame, (x2,y2), (x1,y1), (0, 255, 255), 2)
    # All the results have been drawn on the frame, so it's time to display it.
#     cv2.imshow('Object detector', frame)

    # For Notebook File
    # if count_frame%100 == 5 :
    #     print(count_frame)
    
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        print('break')
        break

# Clean up
video.release()
cv2.destroyAllWindows()