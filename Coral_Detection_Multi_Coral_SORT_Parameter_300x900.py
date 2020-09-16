from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import configparser
import imutils
import time
import os
import cv2
import platform
from sort import *
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing.managers import BaseManager
import edgetpu.detection.engine
from edgetpu.utils import image_processing


from functions.init import tracker__init
from functions.centroidtracker import CentroidTracker
from functions.trackableobject import TrackableObject




def SORT_Matching_func(mot_tracker, seqLength, filename, name_sequmap, config_file_name, p1_counting, p2_counting, Tracking_Distance,
                       InputQueue_rects, InputQueue_totalFrames, InputQueue_Frame, OutputQueueUp, OutputQueueDown,
                       OutputQueueFrame, ):
    trackableObjects = {}
    object_dets = {}
    
    totalUp = 0
    totalDown = 0
    
    with open('Benchmark/res/' + filename + '/' + name_sequmap + "/" + '%s.txt' % (config_file_name),
              'w') as out_file_sort:
        
        while True:
            if not InputQueue_rects.empty():
                object_dets = {}
                rects = InputQueue_rects.get()
                totalFrames = InputQueue_totalFrames.get()
                frame = InputQueue_Frame.get()
                H = frame.shape[1]

                #print("Frames Multi", totalFrames)
                Counter_Up_frame = 0
                Counter_Down_frame = 0
                rects_2 = np.asarray(rects)
                objects_2 = mot_tracker.update(rects_2)

                for i in objects_2:
                    startX = i[0]
                    startY = i[1]
                    endX = i[2]
                    endY = i[3]

                    print(
                        '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                            totalFrames + 1, i[4], startX, startY,
                            endX - startX, endY - startY),
                        file=out_file_sort)

                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    object_dets[i[4]] = (cX, cY, int(i[4]))

                for (objectID, centroid) in object_dets.items():
                    # check to see if a trackable object exists for the current
                    # object ID

                    to = trackableObjects.get(objectID, None)

                    # if there is no existing trackable object, create one
                    if to is None:
                        to = TrackableObject(objectID, centroid)

                    # otherwise, there is a trackable object so we can utilize it
                    # to determine direction
                    else:
                        # the difference between the y-coordinate of the *current*
                        # centroid and the mean of *previous* centroids will tell
                        # us in which direction the object is moving (negative for
                        # 'up' and positive for 'down')
                        y = [c[1] for c in to.centroids]
                        direction = centroid[1] - np.mean(y)
                        to.centroids.append(centroid)

                        # check to see if the object has been counted or not
                        if not to.counted and totalFrames > 1:

                            p1 = np.array([0, p1_counting])
                            p2 = np.array([500, p2_counting])

                            x1 = centroid[0]
                            y1 = centroid[1]

                            p3 = np.array([x1, y1])

                            d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
                            # if the direction is negative (indicating the object
                            # is moving up) AND the centroid is above the center
                            # line, count the object
                            if direction < 0 and d < Tracking_Distance:
                                totalUp += 1
                                Counter_Up_frame += 1
                                to.counted = True
                                cv2.circle(frame, (centroid[0], centroid[1]), 100, (255, 0, 0), -1)

                            # if the direction is positive (indicating the object
                            # is moving down) AND the centroid is below the
                            # center line, count the object
                            elif direction > 0 and d < Tracking_Distance:
                                totalDown += 1
                                Counter_Down_frame += 1
                                to.counted = True
                                cv2.circle(frame, (centroid[0], centroid[1]), 100, (255, 0, 0), -1)

                    # store the trackable object in our dictionary
                    trackableObjects[objectID] = to

                    # draw both the ID of the object and the centroid of the
                    # object on the output frame
                    if to.counted:
                        colour = 255
                    else:
                        colour = 0

                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (colour, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (colour, 255, 0), -1)

                OutputQueueUp.put(Counter_Up_frame)
                OutputQueueDown.put(Counter_Down_frame)
                OutputQueueFrame.put(totalFrames)
                
                info = [
                    ("Up", totalUp),
                    ("Down", totalDown),
                ]

                # loop over the info tuples and draw them on our frame
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                cv2.line(frame, (0, p1_counting), (900, p2_counting), (0, 255, 255), 2)
                cv2.line(frame, (0, p1_counting + Tracking_Distance), (900, p2_counting + Tracking_Distance), (100, 255, 255),2)
                cv2.line(frame, (0, p1_counting - Tracking_Distance), (900, p2_counting - Tracking_Distance), (100, 255, 255),2)
                cv2.imshow("SORT", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or totalFrames >= seqLength:
                    break


def people_counter(export_data, create_Ground_Truth, Display, config_file_name, filename, name_sequmap):
    # filename = "MOT16-03"

    def config_files(config_file_name, filename, create_Ground_Truth):
        config = configparser.ConfigParser()
        config.sections()
        output_vid = None

        suffix = ".mp4"

        path_input = "videos/" + filename + "/" + filename + suffix
        config.read("Benchmark/seqmaps/" + filename + "/SequInfo/" + name_sequmap + "/" + config_file_name + '.ini')
        # output_vid = "output/" + filename + "/" + folder + "/" + filename + suffix
        ground_truth = "videos/" + filename + "/gt/gt.txt"

        # Import Ground truth data
        if create_Ground_Truth:
            gt_dets = np.loadtxt(ground_truth, delimiter=',')  # load gt
        else:
            gt_dets = None

        # SequenceConfig
        Sequence = config['Sequence']
        name = Sequence['name']
        frameRate = int(Sequence['frameRate'])
        frames_per_second_Video = int(Sequence['frames_per_second_Video'])
        seqLength = int(Sequence['seqLength'])
        imWidth = int(Sequence['imWidth'])
        imHeight = int(Sequence['imHeight'])

        # Image
        Image = config['Image']
        crop_image = Image.getboolean('crop_image')
        image_top_crop = int(Image['image_top_crop'])
        image_bottom_crop = int(Image['image_bottom_crop'])
        width_frame = int(Image['width_frame'])

        # DetectionConfig
        DetectionConfig = config['DetectionConfig']
        skip_frames = int(DetectionConfig['skip_frames'])
        confidence_threshold = float(DetectionConfig['confidence_threshold'])
        System_RP = platform.system()
        detection_system = DetectionConfig['detection_system']

        path_model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
        path_protox = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"

        # Counting Line
        CountingLine = config['CountingLine']
        p1_counting = int(CountingLine['p1_counting'])
        p2_counting = int(CountingLine['p2_counting'])
        Tracking_Distance = int(CountingLine['Tracking_Distance'])

        # Tracker
        TrackerConfig = config['TrackerConfig']
        tracker_type = TrackerConfig['tracker_type']

        # ID Matching Sort
        IdMatching = config['ID Matching']
        SORT_Matching = IdMatching.getboolean('SORT_Matching')
        SORT_Max_Age = int(IdMatching['SORT_Max_Age'])
        SORT_Min_Hits = int(IdMatching['SORT_Min_Hits'])

        # ID Matching Euclidean distance
        maxDisappeared_Euc = int(IdMatching['maxDisappeared_Euc'])
        maxDistance_Euc = int(IdMatching['maxDistance_Euc'])

        factor_x = 1
        factor_y = 1
        return name, SORT_Max_Age, SORT_Min_Hits, factor_x, factor_y, path_input, gt_dets, output_vid, frameRate, frames_per_second_Video, seqLength, imWidth, imHeight, Image, crop_image, image_top_crop, image_bottom_crop, width_frame, DetectionConfig, skip_frames, confidence_threshold, System_RP, detection_system, path_model, path_protox, CountingLine, p1_counting, p2_counting, Tracking_Distance, tracker_type, SORT_Matching, maxDisappeared_Euc, maxDistance_Euc

    name, SORT_Max_Age, SORT_Min_Hits, factor_x, factor_y, path_input, gt_dets, output_vid, frameRate, frames_per_second_Video, seqLength, imWidth, \
    imHeight, Image, crop_image, image_top_crop, image_bottom_crop, width_frame, DetectionConfig, skip_frames, \
    confidence_threshold, System_RP, detection_system, path_model, path_protox, CountingLine, p1_counting, \
    p2_counting, Tracking_Distance, tracker_type, SORT_Matching, maxDisappeared_Euc, \
    maxDistance_Euc = config_files(config_file_name, filename, create_Ground_Truth)

    def NN(path_protox, path_model):
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]

        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(path_protox, path_model)

        return CLASSES, net

    def check_detections(results, rects, adding_startX, adding_startY):

        data_out = []

        for obj in results:
            inference = []

            labeltxt = labels[obj.label_id]

            # cv2.rectangle(frame_crop_2, (int(startX), int(startY)), (int(endX), int(endY)), color=(0, 255, 255))

            if labeltxt == "person":
                box = obj.bounding_box.flatten().tolist()
                startX = box[0] + adding_startX
                startY = box[1] + adding_startY
                endX = box[2] + adding_startX
                endY = box[3] + adding_startY
                w_bbox = (endX - startX)
                h_bbox = (endY - startY)
                size = (w_bbox * h_bbox)

                if size > image_bottom_crop and size < image_top_crop:
                #if True:
                    bbox = (startX, startY, w_bbox, h_bbox)
                    # print((startX, startY, endX, endY))
                    cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), color=(0, 255, 255))
                    rects.append((startX, startY, endX, endY))

                    # Export Detections
                    print(
                        '%d,-1,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                        totalFrames + 1, startX * factor_x, startY * factor_y, endX * factor_x - startX * factor_x,
                        endY * factor_y - startY * factor_y), file=out_file_det)

                    tracker_Coral = tracker__init(tracker_type)

                    ok = tracker_Coral.init(frame, bbox)

                    # add the tracker to our list of trackers so it can
                    # utilize it during skip frames
                    trackers.append(tracker_Coral)

                else:
                    cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (100, 255, 255), 10)
                    # print("Size", size)

        return rects

    ####### CORAL ################
    labels_file = 'tpu_models/coco_labels.txt'
    # Read labels from text files. Note, there are missing items from the coco_labels txt hence this!
    labels = [None] * 10
    with open(labels_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split(maxsplit=1)
        labels.insert(int(parts[0]), str(parts[1]))

    engine = edgetpu.detection.engine.DetectionEngine('tpu_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')

    from PIL import Image
    
    mot_tracker = Sort(max_age=SORT_Max_Age, min_hits=SORT_Min_Hits)

        ############## MULTI ############
    maxsize_Queue = 0

    InputQueue_rects = Queue(maxsize=maxsize_Queue)
    InputQueue_totalFrames = Queue(maxsize=maxsize_Queue)
    InputQueue_Frame = Queue(maxsize=maxsize_Queue)
    OutputQueueUp = Queue(maxsize=maxsize_Queue)
    OutputQueueDown = Queue(maxsize=maxsize_Queue)
    OutputQueueFrame = Queue(maxsize=maxsize_Queue)
    rects = None

    p = Process(target=SORT_Matching_func,
                args=(
                    mot_tracker, seqLength, filename, name_sequmap, config_file_name, p1_counting, p2_counting, Tracking_Distance,
                    InputQueue_rects,
                    InputQueue_totalFrames, InputQueue_Frame, OutputQueueUp, OutputQueueDown, OutputQueueFrame,))
    p.daemon = False
    p.start()

    CLASSES, net = NN(path_protox, path_model)

    # Grab a reference to the video file
    print("[INFO] opening video file...")

    vs = cv2.VideoCapture(path_input)

    print(path_input)
    # initialize the video writer (we'll instantiate later if need be)
    writer, W, H, bbox = None, None, None, None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared_Euc, maxDistance_Euc)
    trackers = []

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames, totalDown, totalUp = 0, 0, 0

    # start the frames per second throughput estimator
    fps = FPS().start()
    time.sleep(2)

    start_time = time.time()

    # Einstellung der Realen Frames per Second -> Entsprechend werden Frames geskipped
    frame_count = 1
    frame_skipped = 0
    elapsed = float()

    SORT_Name = "SORT_Output"
    Det_Name = "det"

    if create_Ground_Truth:
        counter_Name = "Counter_GT"
    else:
        counter_Name = "Counter"

    if not os.path.exists("Benchmark/res/" + filename + "/" + name_sequmap):
        os.makedirs("Benchmark/res/" + filename + "/" + name_sequmap)
        os.makedirs("Benchmark/res/" + filename + "/" + name_sequmap + "/Det")
        os.makedirs("Benchmark/res/" + filename + "/" + name_sequmap + "/Counter")

    # loop over frames from the video stream
    with open('Benchmark/res/' + filename + "/" + name_sequmap + '/Det/' + '%s.txt' % (config_file_name),
              'w') as out_file_det, open \
                ('Benchmark/res/' + filename + "/" + name_sequmap + '/Counter/' + '%s.txt' % (config_file_name),
                 'w') as out_file_count:
        while (vs.isOpened()):
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream

            end_time = time.time()
            elapsed = end_time - start_time


            while (frames_per_second_Video * elapsed) >= frame_count:
                
                frame_count += 1
                frame_skipped += 1
                totalFrames += 1
                
                if totalFrames <= seqLength:
                    ret, frame = vs.read()
                    end_time = time.time()
                    elapsed = end_time - start_time


            else:
                frame_count += 1
                
                if totalFrames <= seqLength:
                    ret, frame = vs.read()

            
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q") or totalFrames >= seqLength or ret == False:
                print("Break")
                #p.kill()
                break
            
            (H, W) = frame.shape[:2] 
            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert

            # if the frame dimensions are empty, set them
            

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            #if output_vid is not None and writer is None and export_data:
            #    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            #    writer = cv2.VideoWriter(output_vid, fourcc, 30,
            #                             (W, H), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # initialize Counter per frames
            Counter_Up_frame = 0
            Counter_Down_frame = 0
            
            if ret == True:
                # check to see if we should run a more computationally expensive
                # object detection method to aid our tracker
                if totalFrames % skip_frames == 0 or not all(rects):
                    # set the status and initialize our new set of object trackers
                    status = "Detecting"
                    trackers = []

                    data_out = []
                    rects = []
                    
                    adding_1_startX = 0
                    adding_1_endX = 300
                    adding_1_startY = 0
                    adding_1_endY = 300

                    img_1 = Image.fromarray(frame)
                    results_1 = engine.DetectWithImage(img_1, confidence_threshold, keep_aspect_ratio=True,
                                                       relative_coord=False, top_k=10)
                    
                    rects = check_detections(results_1, rects, adding_1_startX, adding_1_startY)


                else:
                    # loop over the trackers
                    for tracker in trackers:
                        # set the status of our system to be 'tracking' rather
                        # than 'waiting' or 'detecting'
                        status = "Tracking"

                        ok, bbox = tracker.update(frame)

                        pos = bbox

                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

                        if ok:
                            # Tracking success
                            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)

                            # unpack the position object
                            startX = int(bbox[0])
                            startY = int(bbox[1])
                            endX = int(bbox[0] + bbox[2])
                            endY = int(bbox[1] + bbox[3])

                            if export_data:
                                print('%d,-1,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' %
                                      (totalFrames + 1, startX * factor_x, startY * factor_y,
                                       endX * factor_x - startX * factor_x,
                                       endY * factor_y - startY * factor_y),
                                      file=out_file_det)

                            # add the bounding box coordinates to the rectangles list
                            rects.append((startX, startY, endX, endY))
                        else:
                            # Tracking failure
                            cv2.rectangle(frame, p1, p2, (255, 255, 0), 2, 1)

                # use the centroid tracker to associate the (1) old object
                # centroids with (2) the newly computed object centroids

                # SORT_Matching_func(rects, totalFrames, W, p1_counting, p2_counting, Tracking_Distance, frame)

                # rects = []

                InputQueue_totalFrames.put(totalFrames)
                InputQueue_rects.put(rects)
                InputQueue_Frame.put(frame)

                #print("Länge InputQueue", InputQueue_rects.qsize())

                #while OutputQueueUp.empty():
                #    print("Wait")

                if not OutputQueueUp.empty():
                    Counter_Up_frame = OutputQueueUp.get()
                    Counter_Down_frame = OutputQueueDown.get()
                    totalDown += Counter_Down_frame
                    totalUp += Counter_Up_frame
                    frame_Multi = OutputQueueFrame.get()

                    print('%d,%d,%d' % (frame_Multi, Counter_Up_frame, Counter_Down_frame),
                          file=out_file_count)

                # loop over the tracked objects

                # # Export Counter Results
                # print('%d,%d,%d' % (totalFrames, Counter_Up_frame, Counter_Down_frame),
                #       file=out_file_count)
                #
                # # construct a tuple of information we will be displaying on the
                # # frame



                info = [
                    ("Up", totalUp),
                    ("Down", totalDown),
                    ("Status", status),
                ]

                # loop over the info tuples and draw them on our frame
                #for (i, (k, v)) in enumerate(info):
                #    text = "{}: {}".format(k, v)
                #    cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # check to see if we should write the frame to disk
                if writer is not None:
                    writer.write(frame)

                if Display:
                    # show the output frame
                    cv2.imshow("Frame", frame)


                # increment the total number of frames processed thus far and
                # then update the FPS counter
                totalFrames += 1
                fps.update()
                print("Total Frames: ", totalFrames)

    # stop the timer and display FPS information
    fps.stop()
    end_time = time.time()
    elapsed = end_time - start_time
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print("[INFO] approx. FPS Real:", frame_count / elapsed)
    print("[INFO] Skipped Frames:", frame_skipped)
    print("--------------------------------------------------")

    with open('Benchmark/res/' + filename + '/' + name_sequmap + "/" + '%s.txt' % (
            config_file_name + "_frames_per_second"), 'w') as out_file_frames:
        print('FPS Proc: %.2f' % (fps.fps()), file=out_file_frames)
        print('FPS Real: %.2f' % (frame_count / elapsed), file=out_file_frames)
        print('Skip Frs: %.2f' % frame_skipped, file=out_file_frames)
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not input:
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    p.terminate()
    # close any open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':

    filename = "AVG-TownCentre900"
    name_sequmap = "Benchmark_Tracker_Skipped_Frames_Baseline"
    fobj = open("Benchmark/seqmaps/" + filename + "/" + name_sequmap + ".txt", "r")
    export_data = True
    create_Ground_Truth = False
    Display = True

    for line in fobj:
        if not line.rstrip() == "name" and not line.rstrip() == "":
            print(line.rstrip())
            name = line.rstrip()
            # config_file_name = 'seqinfo'
            config_file_name = name
            # filename = "Top-View_1"

            people_counter(export_data, create_Ground_Truth, Display, config_file_name, filename, name_sequmap)


    fobj.close()
