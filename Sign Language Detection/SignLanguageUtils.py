import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
import os

class SignLanguageUtils():
    def __init__(self):
        
        self.mp_holistic = mp.solutions.holistic
        self.mp_model = self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.DATA_PATH = os.path.join('MP_DATA')
        self.actions = np.array(['hello', 'thanks', 'sorry'])
        self.total_sequences = 60
        self.sequence_length = 30

        self.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)

    def landmarksDetection(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.mp_model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return image, results
    
    def drawLandmarks(self, image, results):
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_TESSELATION,
                                       self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                       self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
        
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
 
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

    def extractKeypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])
    
    def structureDataFiles(self, actions, total_sequences, DATA_PATH):
        for action in actions:
            for seq in range(total_sequences):
                try:
                    os.makedirs(os.path.join(DATA_PATH, action, str(seq)))
                except:
                    pass
    
    def addActions(self, action, restructure=True):
        self.actions = np.append(self.actions, action)

        if restructure:
            self.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)
    
    def setVideoLength(self, length):
        self.sequence_length = length
    
    def setNumberVideos(self, number, restructure=True):
        self.total_sequences = number

        if restructure:
            self.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)

    def setDataPath(self, new_path, restructure=True):
        self.DATA_PATH = new_path
        if restructure:
            self.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)
