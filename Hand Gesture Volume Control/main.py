import cv2
import pyautogui
import mediapipe as mp
import math as m

webcam = cv2.VideoCapture(0)
hands_processor = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

x1 = x2 = y1 = y2 = 0

def processHandLandmarks(hand_landmarks):
    global x1, x2, y1, y2
    if hand_landmarks:
        for hand in hand_landmarks:
            drawing_utils.draw_landmarks(image, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    cv2.circle(image, center=(x,y), radius=8, color=(0, 255, 255), thickness=3)
                    x1 = x
                    y1 = y
                if id == 4:
                    cv2.circle(image, center=(x,y), radius=8, color=(0, 255, 255), thickness=3)
                    x2 = x
                    y2 = y
        dist = m.sqrt((x2 - x1)**2 + (y2 - y1)**2) // 4
        cv2.line(image, (x1,y1), (x2,y2), (0, 255, 0), 3)
        if dist > 40:
            pyautogui.press("volumeup")
        else:
            pyautogui.press("volumedown")

while True:
    _ , image = webcam.read()
    frame_height, frame_width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands_output = hands_processor.process(rgb_image)
    hands = hands_output.multi_hand_landmarks

    processHandLandmarks(hands)
    cv2.imshow("Hand Gesture", image)

    if cv2.waitKey(10) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
