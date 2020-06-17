import numpy as np
import os
import pathlib
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from collections import defaultdict
from io import StringIO
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import pygame
from pygame.locals import *

"""Import the object detection module."""
# sys.path.append("..")
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from pyparrot.Minidrone import Mambo

import pyscreenshot as ImageGrab
import time as time
import subprocess
from threading import Timer
import threading
from mss import mss
from PIL import Image
from imutils.video import FPS
import imutils

"""Patches:"""
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1
# Patch the location of gfile
tf.gfile = tf.io.gfile

x2 = 0
y2 = 0
area = 0
#cap = cv2.VideoCapture(0)

def displayWindow():
	#This thread always runs on sideline
    subprocess.run(["gst-launch-1.0", "rtspsrc", "location=rtsp://192.168.99.1/media/stream2", "latency=10" , "!" , "decodebin" , "!" , "xvimagesink"])




"""# Model preparation

## Variables

Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.

By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

## Loader
"""


def load_model(model_name):
    base_url = 'http://download.tensorflow.org/models/object_detection/'
    model_file = model_name + '.tar.gz'
    model_dir = tf.keras.utils.get_file(
        fname=model_name,
        origin=base_url + model_file,
        untar=True)

    model_dir = pathlib.Path(model_dir) / "saved_model"

    model = tf.saved_model.load(str(model_dir))
    model = model.signatures['serving_default']

    return model


"""## Loading label map
Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
"""

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

pygame.init()
screen = pygame.display.set_mode((640,480))
keys = pygame.key.get_pressed()


mamboAddr = "d0:3a:7b:42:e6:20"
mambo = Mambo(mamboAddr, use_wifi=True)
print("trying to connect")
success = mambo.connect(num_retries=3)
print("connected: %s" % success)
# get the state information
print("IMPERIAL COLLEGE LONDON")
print("intelligent Digital Systems Lab (iDSL)")
mambo.smart_sleep(1)
mambo.ask_for_state_update()
mambo.smart_sleep(1)
print("taking off!")
mambo.safe_takeoff(3)

"""# Detection
Load an object detection model:
"""

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

"""Check the model's input signature, it expects a batch of 3-color images of type uint8:"""

# print(detection_model.inputs)

"""And retuns several outputs:"""

# detection_model.output_dtypes
#
# detection_model.output_shapes

"""Add a wrapper function to call the model, and cleanup the outputs:"""


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    output_dict = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # print(' detections class is ', output_dict['detection_classes'])
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the box mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()




    return output_dict


"""Run it on each test image, send commands and show the results:"""


def show_inference(model):
    global x2, y2, command, area
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    #print('Detecting...')
    # image_np = np.array(Image.open(image_path))
    #while True:
        # ret, image_np = cap.read()

        # Actual detection.
    print(cap.shape)
    output_dict = run_inference_for_single_image(model, cap)

        # print(' detections class is ', output_dict['detection_classes'])
        # print(' detections boxes are ', output_dict['detection_boxes'])
        # print(' detections scores are ', output_dict['detection_scores'])
        # Visualization of the results of a detection.

    stream, x, y, object, area = vis_util.visualize_boxes_and_labels_on_image_array(
            cap,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
    # if ((x2 == x) and (y2 == y )):
    #     pass
    # if (object == "person"):
    #
    #     if ((95 <= x <= 310) and (185 <= y <= 305)):  # person moves to right and down
    #         #move left and down
    #         mambo.fly_direct(roll=-5, pitch=0, yaw=0, vertical_movement=-10, duration=0.2)
    #
    #
    #     elif ((311 <= x < 550) and (185 <= y <= 305)):  # person moves to left and down
    #         # move right and down
    #         mambo.fly_direct(roll=5, pitch=0, yaw=0, vertical_movement=-10, duration=0.2)
    #
    #     elif ((95 <= x < 310) and (0 < y <= 184)):  # person moves to right and up
    #         # move left and up
    #         mambo.fly_direct(roll=-5, pitch=0, yaw=0, vertical_movement=10, duration=0.2)
    #
    #     elif ((311 <= x < 550) and (0 < y <= 184)):  # person moves to right and up
    #         # move right and up
    #         mambo.fly_direct(roll=-5, pitch=0, yaw=0, vertical_movement=10, duration=0.2)
    #
    #     elif ((311 <= x < 550) and (185 <= y <= 305)):  # person moves to left
    #         # move right
    #         mambo.fly_direct(roll=5, pitch=0, yaw=0, vertical_movement=0, duration=0.2)
    #
    #     elif (95 <= x < 310):  # person moves to right
    #         # move left
    #         mambo.fly_direct(roll=-5, pitch=0, yaw=0, vertical_movement=0, duration=0.2)




    print("x,y,obejct are ", x, y, object)
    print("area:",area)
    x2 = x
    y2 = y
    return stream,area



def capture_screenshot():
    with mss() as sct:
        monitor = {"top":50, "left":68, "width":638, "height":362}
        sct_img = sct.grab(monitor)
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'RGBX')

displayWindowThread = threading.Thread(target=displayWindow)
displayWindowThread.start()

time.sleep(1)
command = input("Adjust screen and Press Enter to Continue")
frameRate = 0
start_time = time.time()
fps = FPS().start()

while (True):
    img = capture_screenshot()
    cap = np.array(img)  # this is the array obtained from conversion

    pygame.event.pump()
    keys = pygame.key.get_pressed()


    cap,area = show_inference(detection_model)

    if (area < 130000):
        mambo.fly_direct(roll=0, pitch=5, yaw=0, vertical_movement=0, duration=0.2)

    if (keys[K_KP9] == 1):
        break
    cv2.imshow("Live Video", cap)
    cv2.moveWindow("Live Video", 800, 20)  # Use this if gst window opens in bottom
    fps.update()

    if (time.time() >= start_time + 1):
        print("Frame Rate : ", frameRate)
        start_time = time.time()
        frameRate = 0
    else:
        frameRate = frameRate + 1
    if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
print("landing")
mambo.safe_land(3)
mambo.smart_sleep(3)

print("disconnect")
mambo.disconnect()
