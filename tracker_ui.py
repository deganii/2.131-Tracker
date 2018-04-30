import cv2
import sys
from PIL import Image
import numpy as np
import csv

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
WIN_NAME = "2.131 Bacterial Tracker"
POS_TRACKBAR = "FRAME"

class TrackerUI(object):

    def __init__(self, filename, tracker_type='MIL',
        root_path='C:\\dev\\courses\\2.131 - Advanced Instrumentation\\'):
        self._filename = filename
        self._root_path = root_path
        self._tracker_type = tracker_type
        self._cap = cv2.VideoCapture(root_path + filename)
        self._save_video = True
        self.setup_ui()

    def get_tracker(self, tracker_type):
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

    def seek_callback(self, x):
        i = cv2.getTrackbarPos(POS_TRACKBAR, WIN_NAME)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
        _, frame = self._cap.read()
        cv2.imshow(WIN_NAME, frame)

    def skip_frame_generator(self, df):
        cf = self._cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, cf + df)
        cv2.setTrackbarPos(POS_TRACKBAR, WIN_NAME,
            int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = self._cap.read()
        cv2.imshow(WIN_NAME, frame)

    def play_pause(self):
        while True:
            ret, frame = self._cap.read()
            cv2.imshow(WIN_NAME, frame)
            cv2.setTrackbarPos(POS_TRACKBAR, WIN_NAME,
                int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                return

    def setup_ui(self):
        cv2.namedWindow(WIN_NAME)
        cv2.createTrackbar(
            POS_TRACKBAR, WIN_NAME, 0,
            int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.seek_callback)
        # cv2.createButton("Play", self.play, buttonType=cv2.QT_PUSH_BUTTON)

        ret, frame = self._cap.read()
        cv2.imshow(WIN_NAME, frame)

        actions = dict()
        actions[ord("\n")] = lambda: self.track_trajectory()
        actions[ord("\r")] = lambda: self.track_trajectory()
        actions[ord(" ")] = lambda: self.play_pause()
        # actions[ord("D")] = lambda: self.skip_frame_generator(10)
        # actions[ord("d")] = lambda: self.skip_frame_generator(1)
        # actions[ord("a")] = lambda: self.skip_frame_generator(-1)
        # actions[ord("A")] = lambda: self.skip_frame_generator(-10)
        actions[ord("q")] = lambda: exit(0)

        cv2.imshow(WIN_NAME, frame)
        while True:
            key = cv2.waitKey(0) & 0xFF
            def dummy():
                pass
            actions.get(key, dummy)()

    def track_trajectory(self, bbox = (0,0,100,100),
        start_sec=0.0, duration_sec = 10.0, id=1):
        cap = self._cap

        # MIL appears to work best...
        tracker = self.get_tracker(self._tracker_type)

        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame =  int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # int(cap_fps * start_sec)
        duration_frames = int(cap_fps * duration_sec)

        # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        time = (current_frame-1) * cap_fps


        resw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = (resw, resh)

        ret, frame = cap.read()

        if self._save_video:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter(self._root_path + self._filename + '.multi_track1.avi',
                                  fourcc, cap_fps, res)



        # Define an initial bounding box
        # bbox = (287, 23, 86, 320)

        bbox = cv2.selectROI(WIN_NAME, frame, True)
        ok = tracker.init(frame, bbox)

        # Uncomment the line below to select a different bounding box
        # selector_frame = frame.copy()


        csvfile = open(self._root_path + self._filename + '.{0}.csv'.format(id), 'w', newline='')
        bac_writer = csv.writer(csvfile, delimiter=',',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        bac_writer.writerow(['BacteriaId', 'Frame', 'Time (s)', 'X', 'Y'])


        while cap.isOpened() and current_frame < duration_frames:
            ret, frame = cap.read()

            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                bac_writer.writerow([id, current_frame, "{:.3f}".format(time), int(bbox[0] + bbox[2]/2),
                                     int(bbox[1] + bbox[3]/2)])
            else:
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            if self._save_video:
                # Display tracker type on frame
                cv2.putText(frame, "2.131 E.Coli Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
                # Display FPS on frame
                cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);
                out.write(frame)

            # Display result
            cv2.imshow(WIN_NAME, frame)

            current_frame = current_frame + 1
            time = time + 1.0/cap_fps

            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27: break
        csvfile.close()

        if self._save_video:
            out.release()

# Goal - put in start_time seconds, get a screen

#tracker_UI = TrackerUI(Ecoli-Slide-Coverslip.MOV')
tracker_UI = TrackerUI('Ecoli-HangingDrop-20x-Dilution-1.MOV')


# tracker_UI.track_trajectory()

# cap = cv2.VideoCapture(root_path + 'Ecoli-Slide-Coverslip.MOV')
# ret, frame = cap.read()
# mser_multi(frame)