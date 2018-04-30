import cv2
import sys
from PIL import Image
import numpy as np
import csv
import shortuuid
import os
import glob
import re
import ntpath

ROOT_PATH = 'C:\\dev\\courses\\2.131 - Advanced Instrumentation\\'
FILE_NAME = 'Ecoli-HangingDrop-20x-Dilution-1.MOV'

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
WIN_NAME = "2.131 Bacterial Tracker"
POS_TRACKBAR = "FRAME"


class TrackerUI(object):

    def __init__(self, tracker_type='MIL', duration_sec=10.0):
        self._filename = FILE_NAME
        self._root_path = ROOT_PATH
        self._tracker_type = tracker_type
        self._cap = cv2.VideoCapture(ROOT_PATH + FILE_NAME)
        self._cap_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._save_video = True
        self._duration_sec=duration_sec
        self._duration_frames = int(self._cap_fps * self._duration_sec)
        self._is_tracking = False
        self._is_playing = False
        self.load_existing_trackers()
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

    def load_existing_trackers(self):
        self._existing_trackers = {}
        self._trackers_by_id = {}
        self._loaded_trackers = {}
        # self._live_trackers = {}
        tracker_csvs = glob.glob(ROOT_PATH + FILE_NAME + ".*.csv")
        for f in tracker_csvs:
            # parse out key info
            m = re.match('.*\.(\w+)\.(\d+)\.(\d+)\.(\d+)\.csv', ntpath.basename(f))
            if m:
                id = m.group(1)
                w = int(m.group(2))
                h = int(m.group(3))
                start = int(m.group(4))
                # key them by their start frame
                if start in self._existing_trackers:
                    self._existing_trackers[start].append((id,f,w,h))
                else:
                    self._existing_trackers[start] = [(id, f, w, h)]
                self._trackers_by_id[id] = (start, w, h)


    def seek_callback(self, x):
        if self._is_tracking or self._is_playing:
            return
        i = cv2.getTrackbarPos(POS_TRACKBAR, WIN_NAME)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, i-1)
        self.next_frame()

    def skip_frame_generator(self, df):
        cf = self._cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, cf + df)
        cv2.setTrackbarPos(POS_TRACKBAR, WIN_NAME,
            int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = self._cap.read()
        self._current_frame = frame
        cv2.imshow(WIN_NAME, frame)

    def next_frame(self):
        ret, frame = self._cap.read()
        self._current_frame = frame
        # add any existing trackers
        current_frame = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        # load a live
        valid_trackers = [s for s in self._existing_trackers.keys()
                          if s <= current_frame and
                          s + self._duration_frames > current_frame]

        for t in valid_trackers:
            for (id, f, w, h) in self._existing_trackers[t]:
                if id not in self._loaded_trackers:
                    # load the csv
                    with open(f, 'r') as my_file:
                        reader = csv.reader(my_file)
                        self._loaded_trackers[id] = list(reader)
                # now draw the appropriate rect for this tracker
                line = self._loaded_trackers[id][current_frame - t + 1]
                p1 = (int(int(line[3]) - w/2.), int(int(line[4]) - h/2.))
                p2 = (int(int(line[3]) + w/2.), int(int(line[4]) + h/2.))
                cv2.rectangle(frame, p1, p2, (255, 128, 128), 2, 1)
        cv2.imshow(WIN_NAME, frame)
        return frame

    def play_pause(self):
        self.is_playing = True
        while True:
            self.next_frame()
            cv2.setTrackbarPos(POS_TRACKBAR, WIN_NAME,
                int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))
            key = cv2.waitKey(10) & 0xFF
            if key == ord(" "):
                break
        self.is_playing = False

    def setup_ui(self):
        cv2.namedWindow(WIN_NAME)
        cv2.createTrackbar(
            POS_TRACKBAR, WIN_NAME, 0,
            int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.seek_callback)
        # cv2.createButton("Play", self.play, buttonType=cv2.QT_PUSH_BUTTON)
        self.next_frame()

        actions = dict()
        actions[ord("\n")] = lambda: self.track_trajectory()
        actions[ord("\r")] = lambda: self.track_trajectory()
        actions[ord(" ")] = lambda: self.play_pause()
        # actions[ord("D")] = lambda: self.skip_frame_generator(10)
        # actions[ord("d")] = lambda: self.skip_frame_generator(1)
        # actions[ord("a")] = lambda: self.skip_frame_generator(-1)
        # actions[ord("A")] = lambda: self.skip_frame_generator(-10)
        actions[ord("q")] = lambda: exit(0)

        while True:
            key = cv2.waitKey(0) & 0xFF
            def dummy():
                pass
            actions.get(key, dummy)()

    def track_trajectory(self):
        self._is_tracking = True

        id = shortuuid.uuid()

        cap = self._cap

        # MIL appears to work best...
        tracker = self.get_tracker(self._tracker_type)

        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        current_frame =  int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # int(cap_fps * start_sec)
        duration_frames = int(cap_fps * self._duration_sec)

        # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        time = (current_frame-1) * cap_fps


        resw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        resh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        res = (resw, resh)

        frame = self._current_frame
        cv2.putText(frame, "SET MARK", (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow(WIN_NAME, frame)

        if self._save_video:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out_fname = self._root_path + self._filename + \
                        '.multi_track_{0}.avi'.format(id)
            out = cv2.VideoWriter(out_fname, fourcc, cap_fps, res)

        # Define an initial bounding box
        # bbox = (287, 23, 86, 320)

        bbox = cv2.selectROI(WIN_NAME, frame, True)
        ok = tracker.init(frame, bbox)

        # Uncomment the line below to select a different bounding box
        # selector_frame = frame.copy()

        csv_fname = self._root_path + self._filename + \
                    '.{0}.{1}.{2}.{3}.csv'.format(id, bbox[2], bbox[3], current_frame)
        csvfile = open(csv_fname, 'w', newline='')
        bac_writer = csv.writer(csvfile, delimiter=',',
            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        bac_writer.writerow(['BacteriaId', 'Frame', 'Time (s)', 'X', 'Y'])

        invalid = False

        while cap.isOpened() and current_frame < duration_frames:
            # ret, frame = cap.read()
            frame = self.next_frame()


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
            cv2.setTrackbarPos(POS_TRACKBAR, WIN_NAME,
                int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)))

            current_frame = current_frame + 1
            time = time + 1.0/cap_fps

            # stop recording if ESC or space pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 or k == ord(" "):
                invalid = True
                break

        csvfile.close()
        if self._save_video:
            out.release()

        if invalid:
            cv2.putText(frame, "CANCELLED", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75,  (0, 0, 255), 2);
            cv2.imshow(WIN_NAME, frame)
            os.remove(csv_fname)
            os.remove(out_fname)
            cv2.imshow(WIN_NAME, frame)

        self._current_frame = frame
        self._is_tracking = False



# Goal - put in start_time seconds, get a screen

#tracker_UI = TrackerUI(Ecoli-Slide-Coverslip.MOV')
tracker_UI = TrackerUI()


# tracker_UI.track_trajectory()

# cap = cv2.VideoCapture(root_path + 'Ecoli-Slide-Coverslip.MOV')
# ret, frame = cap.read()
# mser_multi(frame)