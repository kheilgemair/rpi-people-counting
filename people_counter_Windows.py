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

from functions.init import tracker__init
from functions.centroidtracker import CentroidTracker
from functions.trackableobject import TrackableObject


def people_counter(export_data, create_Ground_Truth, Display, config_file_name, filename, name_sequmap):
    # filename = "MOT16-03"

    def config_files(config_file_name, filename, create_Ground_Truth):
        config = configparser.ConfigParser()
        config.sections()
        output_vid = None

        suffix = ".mp4"

        path_input = "videos/" + filename + "/" + filename + suffix
        config.read("Benchmark/seqmaps/SequInfo/" + name_sequmap + "/" + config_file_name + '.ini')
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

        # ID Matching Euclidean distance
        maxDisappeared_Euc = int(IdMatching['maxDisappeared_Euc'])
        maxDistance_Euc = int(IdMatching['maxDistance_Euc'])

        factor_x = 1
        factor_y = 1
        return name, factor_x, factor_y, path_input, gt_dets, output_vid, frameRate, frames_per_second_Video, seqLength, imWidth, imHeight, Image, crop_image, image_top_crop, image_bottom_crop, width_frame, DetectionConfig, skip_frames, confidence_threshold, System_RP, detection_system, path_model, path_protox, CountingLine, p1_counting, p2_counting, Tracking_Distance, tracker_type, SORT_Matching, maxDisappeared_Euc, maxDistance_Euc

    name, factor_x, factor_y, path_input, gt_dets, output_vid, frameRate, frames_per_second_Video, seqLength, imWidth, \
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

    CLASSES, net = NN(path_protox, path_model)

    # Grab a reference to the video file
    print("[INFO] opening video file...")

    vs = cv2.VideoCapture(path_input)

    mot_tracker = Sort()

    # initialize the video writer (we'll instantiate later if need be)
    writer, W, H, bbox = None, None, None, None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared_Euc, maxDistance_Euc)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames, totalDown, totalUp = 0, 0, 0

    # start the frames per second throughput estimator
    fps = FPS().start()

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
    with open('Benchmark/res/' + filename + '/' + name_sequmap + "/" + '%s.txt' % (config_file_name),
              'w') as out_file_sort, open \
                ('Benchmark/res/' + filename + "/" + name_sequmap + '/Det/' + '%s.txt' % (config_file_name),
                 'w') as out_file_det, open \
                ('Benchmark/res/' + filename + "/" + name_sequmap + '/Counter/' + '%s.txt' % (config_file_name),
                 'w') as out_file_count:
        while True:
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            time.sleep(0.001)
            end_time = time.time()
            elapsed = end_time - start_time

            while (frames_per_second_Video * elapsed) >= frame_count:
                frame_count += 1
                frame_skipped += 1
                frame = vs.read()
                frame = frame[1] if input else frame
                end_time = time.time()
                elapsed = end_time - start_time
            else:
                frame_count += 1
                frame = vs.read()
                frame = frame[1] if input else frame
                if crop_image:
                    frame = frame[image_top_crop:image_bottom_crop, :]
                    frame_small = imutils.resize(frame, width_frame)
                    factor_x = frame.shape[0] / frame_small.shape[0]
                    factor_y = frame.shape[1] / frame_small.shape[1]
                    frame = frame_small

                    (H, W) = frame.shape[:2]
                    cv2.line(frame, (0, p1_counting), (W, p2_counting), (0, 255, 255), 2)
                else:
                    frame_small = imutils.resize(frame, width_frame)
                    factor_x = frame.shape[0] / frame_small.shape[0]
                    factor_y = frame.shape[1] / frame_small.shape[1]
                    frame = frame_small
                    frame = imutils.resize(frame, width_frame)

                    (H, W) = frame.shape[:2]
                    p1_counting = H // 2
                    p2_counting = H // 2
                    cv2.line(frame, (0, p1_counting), (W, p2_counting), (0, 255, 255), 2)

            cv2.line(frame, (0, p1_counting + Tracking_Distance), (W, p2_counting + Tracking_Distance), (100, 255, 255),
                     2)
            cv2.line(frame, (0, p1_counting - Tracking_Distance), (W, p2_counting - Tracking_Distance), (100, 255, 255),
                     2)
            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert

            # if the frame dimensions are empty, set them
            (H, W) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if output_vid is not None and writer is None and export_data:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_vid, fourcc, 30,
                                         (W, H), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            status = "Waiting"
            rects = []

            # initialize Counter per frames
            Counter_Up_frame = 0
            Counter_Down_frame = 0

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if totalFrames % skip_frames == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers = []
                if not create_Ground_Truth:
                    # convert the frame to a blob and pass the blob through the
                    # network and obtain the detections

                    if platform.system() == "Windows":
                        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                        net.setInput(blob)
                        detections = net.forward()

                    # loop over the detections
                    for i in np.arange(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated
                        # with the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections by requiring a minimum
                        # confidence
                        if confidence > confidence_threshold:
                            # extract the index of the class label from the
                            # detections list
                            idx = int(detections[0, 0, i, 1])

                            # if the class label is not a person, ignore it
                            if CLASSES[idx] != "person":
                                continue

                            # compute the (x, y)-coordinates of the bounding box
                            # for the object
                            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                            (startX, startY, endX, endY) = box.astype("int")

                            bbox = (startX, startY, (endX - startX), (endY - startY))

                            if export_data:
                                print('%d,-1,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                                    totalFrames + 1, startX * factor_x, startY * factor_y,
                                    endX * factor_x - startX * factor_x, endY * factor_y - startY * factor_y),
                                      file=out_file_det)

                            # Draw a green bounding box
                            cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (0, 255, 0), 2)

                            # add the bounding box coordinates to the rectangles list
                            rects.append((startX, startY, endX, endY))

                            tracker = tracker__init(tracker_type)

                            ok = tracker.init(frame, bbox)

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            trackers.append(tracker)

                # Import Ground truth vom gt_dets
                else:
                    dets_ = gt_dets[gt_dets[:, 0] == (totalFrames + 1), 1:6]
                    for det in dets_:
                        tracker = tracker__init(tracker_type)
                        det[1] = int(det[1] / factor_x)
                        det[2] = int(det[2] / factor_y)
                        det[3] = int(det[3] / factor_x)
                        det[4] = int(det[4] / factor_y)
                        bbox = (det[1], det[2], det[3], det[4])
                        ok = tracker.init(frame, bbox)
                        trackers.append(tracker)
                        cv2.rectangle(frame, (int(det[1]), int(det[2]), int(det[3]), int(det[4])), (0, 255, 0), 2)
                        det[3] += det[1]  # convert from w to x2
                        det[4] += det[2]  # convert from h to y2
                        rects.append((det[1], det[2], det[3], det[4]))


            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
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

            object_dets = {}

            if SORT_Matching:
                rects_2 = np.asarray(rects)
                objects_2 = mot_tracker.update(rects_2)
                for i in objects_2:
                    startX = i[0]
                    startY = i[1]
                    endX = i[2]
                    endY = i[3]

                    if export_data:
                        print(
                            '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                                totalFrames + 1, i[4], startX * factor_x, startY * factor_y,
                                endX * factor_x - startX * factor_x, endY * factor_y - startY * factor_y),
                            file=out_file_sort)

                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    object_dets[i[4]] = (cX, cY, int(i[4]))

                objects = object_dets
            else:
                objects = ct.update(rects)

            if export_data and not SORT_Matching:
                for (i, (startX, startY, endX, endY)) in enumerate(rects):
                    # use the bounding box coordinates to derive the centroid
                    cX = int((startX + endX) / 2.0)
                    cY = int((startY + endY) / 2.0)
                    inputCentroids = np.array([cX, cY])

                    for (objectID, centroid) in objects.items():
                        if np.array_equal(centroid, inputCentroids):
                            print(
                                '%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
                                    totalFrames + 1, (objectID + 1), startX * factor_x, startY * factor_y,
                                    endX * factor_x - startX * factor_x, endY * factor_y - startY * factor_y),
                                file=out_file_sort)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
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
                    if not to.counted and totalFrames > 0:

                        p1 = np.array([0, p1_counting])
                        p2 = np.array([W, p2_counting])

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

            # Export Counter Results
            print('%d,%d,%d' % (totalFrames, Counter_Up_frame, Counter_Down_frame),
                  file=out_file_count)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Up", totalUp),
                ("Down", totalDown),
                ("Status", status),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            if Display:
                # show the output frame
                cv2.imshow("Frame", frame)

            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q") or totalFrames == seqLength:
                break

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

    print("--------------------------------------------------")
    # print("[INFO] approx. FPS Real:", frame_count / elapsed)
    # print("[INFO] Skipped Frames:", frame_skipped)

    with open('Benchmark/res/' + filename + '/' + name_sequmap + "/" + '%s.txt' % (
            config_file_name + "_frames_per_second"), 'w') as out_file_frames:
        print('%.2f' % (fps.fps()), file=out_file_frames)
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()

    # if we are not using a video file, stop the camera video stream
    if not input:
        vs.stop()

    # otherwise, release the video file pointer
    else:
        vs.release()

    # close any open windows
    cv2.destroyAllWindows()


if __name__ == '__main__':

    name_sequmap = "Benchmark_Matching"
    fobj = open("Benchmark/seqmaps/" + name_sequmap + ".txt", "r")

    export_data = True
    create_Ground_Truth = True
    Display = True

    for line in fobj:
        if not line.rstrip() == "name" and not line.rstrip() == "":
            print(line.rstrip())
            name = line.rstrip()
            # config_file_name = 'seqinfo'
            config_file_name = name
            filename = "AVG-TownCentre"

            people_counter(export_data, create_Ground_Truth, Display, config_file_name, filename, name_sequmap)

    fobj.close()
