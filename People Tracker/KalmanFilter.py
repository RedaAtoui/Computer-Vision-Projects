import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, x, y, filter_id):
        
        self.kalman_filter = cv2.KalmanFilter(4, 2)
        self.filter_id = filter_id

        # Assuming constant velocity
        self.kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                        [0, 1, 0, 1], 
                                                        [0, 0, 1, 0],
                                                        [0, 0, 0, 1]], dtype=np.float32)
        
        self.kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                        [0, 1, 0, 0]], dtype=np.float32)
        
        self.kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        self.kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)

        self.kalman_filter.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)


    def predict(self):
        predicted = self.kalman_filter.predict()
        return int(predicted[0]), int(predicted[1])
    
    def update(self, x, y):
        feedback_measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman_filter.correct(measurement=feedback_measurement)

    def getFilterID(self):
        return self.filter_id