import cv2
import numpy as np
from ultralytics import YOLO
from KalmanFilter import KalmanFilter
import random
from tensorflow import keras
from mtcnn import MTCNN

class PeopleDetector:
    DISABLED = -1
    ALL = 0
    CLASSIFIED = 1

    FEMALE = 0
    MALE = 1
    
    def __init__(self, classifying_mode=False, tracking_mode=False, gender=FEMALE):
        
        self.yolo_model = YOLO("yolov8s.pt")
        self.people_tracked = []
        self.trackers = []
        self.track_ids = {}
        self.gender_labels = ['female', 'male']

        self.gender_tracked = gender

        if not tracking_mode:
            self.tracking = self.DISABLED
        else:
            if classifying_mode:
                self.tracking = self.CLASSIFIED
                self.gender_classifier = keras.models.load_model("C:\\Users\\USER\\Documents\\Work\\Models\\gender_classifier_model.h5")
                self.face_cascade = cv2.CascadeClassifier('C:\\Users\\USER\\Documents\\Work\\Oculi preps\\Face Detection\\haarcascade_frontalface_default.xml')
                self.face_detector = MTCNN()
 
            else:
                self.tracking = self.ALL
                self.gender_classifier = None
                self.face_cascade = None
                self.face_detector = None

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

    def detectGender(self, person_image):

        # if using the face cascade haar method
        # gray_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
        # faces = self.face_cascade.detectMultiScale(gray_image, 1.7, 4, minSize=(30, 30))
        

        # when using the MTCNN method
        faces_boxes = self.detectFaces(person_image)

        if len(faces_boxes) == 0:
            return None

        for (xf, yf, wf, hf) in faces_boxes:
            face_img = person_image[yf:yf + hf, xf:xf + wf]
    
            face_img = cv2.resize(face_img, (128, 128))
            face_img = face_img / 255.0
            face_img = np.expand_dims(face_img, axis=0)
        
            prediction = self.gender_classifier.predict(face_img)
            predicted_gender = self.gender_labels[int(prediction[0] > 0.5)]

            return predicted_gender

    def detectFaces(self, frame):
        results = self.face_detector.detect_faces(frame)
        faces = [result["box"] for result in results if result["confidence"] > 0.7]

        return faces

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

            results = self.yolo_model.track(frame, persist=True)

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

                            if self.tracking != self.DISABLED and track_id is not None:
                                if self.tracking == self.ALL:
                                    if track_id not in self.people_tracked:
                                        # self.people_tracked.append(track_id)
                                        self.createTracker(track_id, ((x1 + x2) // 2, (y1 + y2) // 2))

                                elif self.tracking == self.CLASSIFIED:
                                    person_image = frame[y1:y2, x1:x2]
                                    predicted_gender = self.detectGender(person_image)
                                    
                                    if predicted_gender is not None:
                                        if self.gender_tracked == self.FEMALE and predicted_gender == 'female':
                                            if track_id not in self.people_tracked:
                                                self.createTracker(track_id, ((x1 + x2) // 2, (y1 + y2) // 2))

                                        elif self.gender_tracked == self.MALE and predicted_gender == 'male':
                                            if track_id not in self.people_tracked:
                                                self.createTracker(track_id, ((x1 + x2) // 2, (y1 + y2) // 2))

                                x_predict, y_predict = self.updateTrackers(track_id, measured_x=(x1 + x2) // 2, measured_y=(y1 + y2) // 2)
                                if x_predict > 0 and y_predict > 0:
                                    b, g, r = self.track_ids[track_id]
                                    cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), 2)
                                    cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (b, g, r), 2)
                                    cv2.circle(frame, (x_predict, y_predict), 5, (b, g, r), -1)
                            
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(frame, f'Person {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            out.write(frame)
                    
        self.cap.release()
        out.release()

# MORE STEPS CAN BE DONE LIKE MAKE IT A GUI FOR BETTER USER EXPERIENCE AND FEEDING DIRECTORY FILES MORE DYNAMICALLY
hello = PeopleDetector(classifying_mode=True, tracking_mode=True, gender=1)
hello.detectPeople(video_source_path="C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\people_Test.mp4", 
                   video_target_path="C:\\Users\\USER\\Documents\\Work\\Oculi preps\\People Tracker\\output")