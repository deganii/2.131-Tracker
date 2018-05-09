import cv2
import sys
from PIL import Image
import numpy as np
import csv

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
root_path = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\'

def mser_multi(frame):
    mser = cv2.MSER_create(
        _delta = 4,
        _min_area = 100,
        _max_area = 500,
        _max_variation = 15.0,
        _min_diversity = 10.0,
        _max_evolution = 10,
        _area_threshold = 12.0,
        _min_margin = 2.9,
        _edge_blur_size = 5)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()
    regions, q = mser.detectRegions(gray)

    #polylines
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))

    #boundingboxes
    mask = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)
    mask = cv2.dilate(mask, np.ones((150,150), np.uint8))

    for contour in hulls:
        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    bboxes = []
    for i, contour in enumerate(hulls):
        x,y,w,h = cv2.boundingRect(contour)
        if(x < 640 and y < 480):
            bboxes.append(cv2.boundingRect(contour))
        print(x,y,w,h) #print coordinates and dimensions of bounding boxes

    # cv2.imshow('img', vis)
    # cv2.imshow('img2', gray)
    # cv2.waitKey()
    expanded_bboxes = []
    for bbox in bboxes:
        # bacteria can rotate so get the largest dimension square
        x,y,w,h = bbox
        if(w > h):
            dh = w - h
            h = w
            y = max(0, int(y - dh/2))
        else:
            dw = h - w
            w = h
            x = max(0, int(x - dw/2))
        expanded_bboxes.append((x,y,w,h))
    nonoverlapping_bboxes = non_max_suppression_fast(expanded_bboxes, 0.2)
    return nonoverlapping_bboxes

"""
    Fast non-maximum suppression algorithm by Malisiewicz et al.
    @param boxes: Array of boundary boxes to perform non-max suppression on.
    @param overlapThresh: Acceptable overlap threshold. 0<=overlapTresh<=1.
"""
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes)

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    boxes = boxes[pick].astype("int")
    t_boxes = []
    for box in boxes:
        x,y,w,h = box
        t_boxes.append((x, y, w, h))
    return t_boxes



def get_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()

def track_multi(filename, res = (1920, 1080),
                start_sec=0.0, duration_sec = 10.0):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']

    # MIL appears to work best...
    tracker_type = tracker_types[1]

    # tracker = cv2.Tracker_create(tracker_type)
    multi_tracker = cv2.MultiTracker_create()

    # cap = cv2.VideoCapture(root_path + 'Rhodosprillum-HangingDrop-3ul.MOV')
    cap = cv2.VideoCapture(root_path + filename)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = int(cap_fps * start_sec)
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    time = current_frame * cap_fps

    ret, frame = cap.read()
    resw, resh = res
    frame = frame[:resh, :resw, ...]

    # run MSER
    # bboxes = mser_multi(frame)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(root_path + filename + '.multi_track1.avi',
                          fourcc, cap_fps, res )

    # Define an initial bounding box
    # bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    selector_frame = frame.copy()
    bboxes = []
    for i in range(20):
        bbox = cv2.selectROI(selector_frame, False)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(selector_frame, p1, p2, (255, 0, 0), 2, 1)
        bboxes.append(bbox)
        # k = cv2.waitKey() & 0xff
        # if k == 27: break

    # Initialize tracker with first frame and bounding box
    for bbox in bboxes:
        # ok = tracker.init(frame, bbox)
        tracker = get_tracker(tracker_type)
        ok = multi_tracker.add(tracker, frame, bbox)

    csvfile = open(root_path + filename + '.csv', 'w', newline='')
    bac_writer = csv.writer(csvfile, delimiter=',',
        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    bac_writer.writerow(['BacteriaId', 'Frame', 'Time (s)', 'X', 'Y'])
    tracker_len = 0


    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[:resh, :resw, ...]
        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bboxes = multi_tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            for idx, bbox in enumerate(bboxes):
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                bac_writer.writerow([idx, current_frame, "{:.3f}".format(time), int(bbox[0] + bbox[2]/2),
                                     int(bbox[1] + bbox[3]/2)])
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "2.131 Bacterial Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # array_alpha = np.array([2.0])
        # array_beta = np.array([0.0])
        #

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # cv2.add(frame,array_beta, frame)
        # multiply every pixel value by alpha
        # cv2.multiply(frame,array_alpha, frame)

        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)
        # out.write(frame)

        # Display result
        cv2.imshow("Tracking", frame)

        current_frame = current_frame + 1
        time = time + 1.0/cap_fps

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    csvfile.close()
    out.release()

# Goal - put in start_time seconds, get a screen

# track_multi('Ecoli-Slide-Coverslip.MOV')
track_multi('Ecoli-HangingDrop-20x-Dilution-1.MOV', res = (1920, 768))

# cap = cv2.VideoCapture(root_path + 'Ecoli-Slide-Coverslip.MOV')
# ret, frame = cap.read()
# mser_multi(frame)