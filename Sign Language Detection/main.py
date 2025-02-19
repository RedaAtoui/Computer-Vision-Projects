from SignLanguageUtils import SignLanguageUtils
import cv2
from tensorflow import keras
import numpy as np

sign_language_utils = SignLanguageUtils()

threshold = 0.6
video = []
sentence = []

actions = sign_language_utils.actions
total_sequences = sign_language_utils.total_sequences
sequence_length = sign_language_utils.sequence_length
DATA_PATH = sign_language_utils.DATA_PATH

mp_model = sign_language_utils.mp_model
sign_language_model = keras.models.load_model("C:\\Users\\USER\\Documents\\Work\\Oculi preps\\Sign Language Detection\\sign_language_detector.h5")

cap = cv2.VideoCapture(0)
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

def probibalityViz(res, input_frame):
    output_frame = input_frame.copy()
    for index, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+index*40), (int(prob*100), 90+index*40), colors[index], -1)
        cv2.putText(output_frame, actions[index], (0, 85+index*40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame

while cap.isOpened():
    ret, frame = cap.read()
    image, results = sign_language_utils.landmarksDetection(frame)
    sign_language_utils.drawLandmarks(image, results)
    
    keypoints = sign_language_utils.extractKeypoints(results=results)
    video.insert(0, keypoints)
    video = video[:30]

    if len(video) == 30:
        res = sign_language_model.predict(np.expand_dims(video, axis=0))[0]
        predicted_action = actions[np.argmax(res)]

        if res[np.argmax(res)] > threshold:
            if len(sentence) > 0:
                if predicted_action != sentence[-1]:
                    sentence.append(predicted_action)
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