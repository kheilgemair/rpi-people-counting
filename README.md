# rpi-people-counting
Image-based person counter using RaspberryPI, Coral and machine leaning.


In this concept, the camera images are processed on a RaspberryPi by tensorflow and analyzed for the number of people in an image. 
The image processing is done via a Google Coral using the MobileNetSSD v2 (COOC). After the persons in the image are recognized, they are tracked by the SORT algorithm. Also included is an initialization file, which can also be used by the Matlab benchmark program to evaluate the results directly.


# Prerequisites

## Hardware:
Raspberry Pi  <br>
Raspberry Pi Camera  <br>
Google Coral  <br>


## Dependencies:

This code makes use of the following packages:
1. [`scikit-learn`](https://scikit-learn.org)
2. [`scikit-image`](https://scikit-image.org/)
3. [`FilterPy`](https://filterpy.readthedocs.io)
4. [`imutils`](https://pypi.org/project/imutils/)
5. [`numpy`](https://numpy.org/)
6. [`configparser`](https://docs.python.org/3/library/configparser.html)
7. [`time`](https://docs.python.org/3/library/time.html)
8. [`os`](https://docs.python.org/3/library/os.html)
9. [`cv2`](https://opencv.org/)
10. [`platform`](https://docs.python.org/3/library/platform.html)
11. [`multiprocessing`](https://pypi.org/project/multiprocess/)
12. [`edgetpu`](https://coral.ai/)

# Architecture
Due to the modular design, different configurations can be tested very easily. The SequMap defines the individual sequences for a video. The corresponding configurations are stored in SequInfo. All configurations are then processed by the Python program. By integrating the SequMap into the Matlab Evaluation program, the results can then be analyzed very quickly and easily.

![Title](https://github.com/tum-gis/rpi-people-counting/blob/master/Workflow%20Diagram.png?raw=true)

## INI File SequInfo

The folder SequInfo contains the corresponding initialization files for SequMaps. The parameters of this file are presented below.


#### [Sequence]
name = AVG-Town  <br>
frameRate = 0  <br>
frames_per_second_Video = 0  <br>
seqLength = 1500  <br>
imWidth = 1920  <br>
imHeight = 1080  <br>

The `[Sequence]` settings define the basic parameters of the video. With `seqLength` the frame number at which the program ends is specified. 

#### [Image]
crop_image = False  <br>
image_top_crop = 200  <br>
image_bottom_crop = 800  <br>
width_frame = 1200  <br>

The `[Image]` can be used to edit the frame before analyzing the CNN. 


#### [DetectionConfig]
skip_frames = 1 <br>
confidence_threshold = 0.3 <br>

With `[DetectionConfig]` the detection can be adjusted. `skip_frames` determines the number of frames when switching between detection and tracker. A value of 1 means that a tracker and CNN are used alternately. With a value of 5, first the CNN is used and then a tracker for 5 frames. `confidence_threshold` specifies the threshold at which detections are retained. 

#### [Background]
history = 400 <br>
varThreshold = 16 <br>
detectShadows = False <br>
width_Background = 600 <br>
adding = 50 <br>

Additionally, a background subtraction was tested. Unfortunately, this could not improve performance, so it is not used in this implementation. 

#### [CountingLine]
p1_counting = 150 <br>
p2_counting = 150 <br>
Tracking_Distance = 50 <br>

By the setting `[CountingLine]`, the counting can be adjusted. `p1_counting` and `p2_counting` indicate the height of the counting line on the Y-axis. `Tracking_Distance` defines the distance from the counting line at which the objects are detected. 

#### [TrackerConfig]
tracker_type = KCF <br>

`[TrackerConfig]` defines the tracker in operation. Available trackers are `KCF`, `MOSSE` and `MedianFlow`.

#### [ID Matching]
SORT_Matching = False <br>
maxDisappeared_Euc = 40 <br>
maxDistance_Euc = 50 <br>
SORT_Max_Age = 2 <br>
SORT_Min_Hits = 3 <br> 

The setting `[ID Matching]` allows switching between two different ID Matching systems like Eucleadean Distance and SORT. 


# Links for further informations

[Tensorflor Edge TPU](https://coral.ai/)

[The Multiple Object Tracking Benchmark](https://motchallenge.net/)

[Simple online and realtime tracking algorithm](https://github.com/abewley/sort)

