import cv2
import sys
from PIL import Image
import numpy as np
import csv


(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def track_ecoli(regions):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    # MIL appears to work best...
    tracker_type = tracker_types[1]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
    root_path = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\'
    img_path = root_path + 'E.coli.tif'
    pil_img = Image.open(img_path)
    frame = np.array(pil_img)
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(root_path + 'ecoli_track1.avi',
                          fourcc, 5.0,(640,480) )

    # Define an initial bounding box
    bbox = (287, 23, 86, 320)

    # Uncomment the line below to select a different bounding box
    bbox = cv2.selectROI(frame, False)

    # Initialize tracker with first frame and bounding box
    ok = tracker.init(frame, bbox)

    csvfile = open(root_path + 'E.coli1.csv', 'w', newline='')
    bac_writer = csv.writer(csvfile, delimiter=',',
        quotechar='|', quoting=csv.QUOTE_MINIMAL)

    bac_writer.writerow(['BacteriaId', 'Time', 'X', 'Y'])
    for i in range(149):
        # Read a new frame
        pil_img.seek(i)
        frame = np.array(pil_img)

        # Start timer
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox = tracker.update(frame)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

            bac_writer.writerow([1, i, int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)])
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Display tracker type on frame
        cv2.putText(frame, "2.131 E.Coli Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

        frame = frame[:480, :640, ...]

        array_alpha = np.array([1.45])
        array_beta = np.array([-150.0])

        cv2.add(frame,array_beta, frame)
        # multiply every pixel value by alpha
        cv2.multiply(frame,array_alpha, frame)

        out.write(frame)
        # out.write(frame)

        # Display result
        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    csvfile.close()
    out.release()

track_ecoli(None)