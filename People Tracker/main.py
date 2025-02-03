import cv2
import numpy as np
from ultralytics import YOLO


class PeopleDetector():
    def __init__(self, video_path=None):
        
        self.model = YOLO("yolov8s.pt")

        # TODO: MOST PROB NO NEED FOR INSTANCE VARIABLES
        self.video_path = video_path
        self.cap = None

    def createBoundingBox(self, frame):
        return

    def setVideoPath(self, path):
        self.video_path = path

    def detectPeople(self, video_source_path, video_target_path):
        
        self.cap = cv2.VideoCapture(video_source_path)
        
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

        while self.cap.isOpened():
            recieved, frame = self.cap.read()
            if not recieved:
                break

            results = self.model(frame)

            for result in results:
                if hasattr(result, 'boxes'):
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0].item()
                        class_id = int(box.cls[0])
                        if class_id == 0 and confidence > 0.6:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(frame)
        self.cap.release()
        out.release()

hello = PeopleDetector()
hello.detectPeople(video_source_path="C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\people.mp4", video_target_path="C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\")