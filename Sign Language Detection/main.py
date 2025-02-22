from SignLanguageUtils import SignLanguageUtils
import cv2
from tensorflow import keras
import numpy as np
import socket
import subprocess
import time

cpp_TTS = "SignLanguage_TTS\\out\\build\\x64-debug\\SignLanguage_TTS.exe"
subprocess.Popen(cpp_TTS, shell=True)
time.sleep(2)


HOST = "127.0.0.1"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

sign_language_utils = SignLanguageUtils()

threshold = 0.7
video = []
sentence = []
predictions = []

actions = sign_language_utils.actions
total_sequences = sign_language_utils.total_sequences
sequence_length = sign_language_utils.sequence_length
DATA_PATH = sign_language_utils.DATA_PATH

mp_model = sign_language_utils.mp_model
sign_language_model = keras.models.load_model("sign_language_detector.h5")

cap = cv2.VideoCapture(0)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

def probibalityViz(res, input_frame):
    output_frame = input_frame.copy()
    for index, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+index*40), (int(prob*100), 90+index*40), colors[index], -1)
        cv2.putText(output_frame, actions[index], (0, 85+index*40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame



conn, addr = server.accept()
while cap.isOpened():
    ret, frame = cap.read()
    image, results = sign_language_utils.landmarksDetection(frame)
    sign_language_utils.drawLandmarks(image, results)
    
    keypoints = sign_language_utils.extractKeypoints(results=results)
    video.append(keypoints)
    video = video[-30:]

    if len(video) == 30:
        res = sign_language_model.predict(np.expand_dims(video, axis=0))[0]
        predicted_action = actions[np.argmax(res)]
        predictions.append(predicted_action)

        if np.unique(predictions[-10:])[0] == predicted_action:
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if predicted_action != sentence[-1]:
                        sentence.append(predicted_action)
                        conn.sendall(predicted_action.encode('utf-8'))
                else:
                    sentence.append(predicted_action)
        
        if len(sentence) > 5:
            sentence = sentence[-5:]

        image = probibalityViz(res, image)
    cv2.rectangle(image, (0, 0), (640, 40), (247, 117, 16), -1)
    cv2.putText(image, " ".join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow("", image)
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
server.close()