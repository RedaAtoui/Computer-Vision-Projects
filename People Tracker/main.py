import cv2
import numpy as np
from ultralytics import YOLO
from KalmanFilter import KalmanFilter
import random

class PeopleDetector:
    DISABLED = -1
    ALL = 0
    CLASSIFIED = 1
    
    def __init__(self, classifying_mode=False, tracking_mode=False):
        
        self.model = YOLO("yolov8s.pt")
        self.people_tracked = []
        self.trackers = []
        self.track_ids = {}

        if not tracking_mode:
            self.tracking = self.DISABLED
        else:
            if classifying_mode:
                self.tracking = self.CLASSIFIED
            else:
                self.tracking = self.ALL

    def createTracker(self, tracker_id, bbox_center):        
        tracker = KalmanFilter(x=bbox_center[0], y=bbox_center[1], filter_id=tracker_id)
        self.trackers.append(tracker)
    
    def updateTrackers(self, tracker_id, measured_x, measured_y):      
        for tracker in self.trackers:
            if tracker_id == tracker.getFilterID():
                x_predict, y_predict = tracker.predict()
                tracker.update(measured_x, measured_y)
                return x_predict, y_predict
        return -1, -1
    
    def detectPeople(self, video_source_path, video_target_path):
        
        self.cap = cv2.VideoCapture(video_source_path)
        
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(f'{video_target_path}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

        while self.cap.isOpened():
            recieved, frame = self.cap.read()
            if not recieved:
                break

            results = self.model.track(frame, persist=True)

            for result in results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0])
                        track_id = int(box.id[0].item()) if box.id is not None else None

                        if class_id == 0 and confidence > 0.7:
                            if track_id not in list(self.track_ids.keys()):
                                self.track_ids[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                            if track_id is not None:
                                if self.tracking == self.ALL:
                                    if track_id not in self.people_tracked:
                                        # self.people_tracked.append(track_id)
                                        self.createTracker(track_id, ((x1 + x2) // 2, (y1 + y2) // 2))

                                x_predict, y_predict = self.updateTrackers(track_id, measured_x=(x1 + x2) // 2, measured_y=(y1 + y2) // 2)
                                if x_predict > 0 and y_predict > 0:
                                    b, g, r = self.track_ids[track_id]
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)
                                    cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 2)
                                    cv2.circle(frame, (x_predict, y_predict), 5, (b, g, r), -1)

                            # TODO: SOLVE FOR OTHER CONDITIONS OF TRACKING

            out.write(frame)
        self.cap.release()
        out.release()

hello = PeopleDetector(classifying_mode=False, tracking_mode=True)
hello.detectPeople(video_source_path="C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\people_14.mp4", 
                   video_target_path="C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\output")
