from SignLanguageUtils import SignLanguageUtils
import cv2
import os
import numpy as np

class SignLanguageDataCollector():
    def __init__(self, video_path=0):

        self.sign_language_utils = SignLanguageUtils()
        self.video_path = video_path

        self.actions = self.sign_language_utils.actions
        self.total_sequences = self.sign_language_utils.total_sequences
        self.sequence_length = self.sign_language_utils.sequence_length
        self.DATA_PATH = self.sign_language_utils.DATA_PATH

    def gatherData(self):
        cap = cv2.VideoCapture(self.video_path)

        for action in self.actions:
            for vid_index in range(self.total_sequences):
                for frame_index in range(self.sequence_length):

                    ret, frame = cap.read()
                    image, results = self.sign_language_utils.landmarksDetection(frame)
                    self.sign_language_utils.drawLandmarks(image, results)

                    if frame_index == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting {} frames for Video number {} '.format(action, vid_index), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.imshow("Progress", image)
                        cv2.waitKey(2000)
                    else:
                        cv2.putText(image, 'Collecting {} frames for Video number {} '.format(action, vid_index), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                        
                        cv2.imshow("Progress", image)
                    
                    landmark_keypoints = self.sign_language_utils.extractKeypoints(results=results)
                    np_path = os.path.join(self.DATA_PATH, action, str(vid_index), str(frame_index))
                    np.save(np_path, landmark_keypoints)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if cv2.waitKey(10) == 27:
                        break
        
        cap.release()
        cv2.destroyAllWindows()

    def addActions(self, action):
        self.actions = np.append(self.actions, action)
        self.sign_language_utils.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)
    
    def setNumberVideos(self, number):
        self.total_sequences = number
        self.sign_language_utils.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)

    def setDataPath(self, new_path):
        self.DATA_PATH = new_path
        self.sign_language_utils.structureDataFiles(self.actions, self.total_sequences, self.DATA_PATH)

    def setVideoLength(self, length):
        self.sequence_length = length
    
    def setVideoPath(self, vid_path):
        self.video_path = vid_path

data_collector = SignLanguageDataCollector()
data_collector.gatherData()