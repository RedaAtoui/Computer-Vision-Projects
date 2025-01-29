import cv2
import mediapipe as mp
import pyautogui

webcam = cv2.VideoCapture(0)
face_mesh_landmarks = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
drawing_utils = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

def processFaceLandmarks(face_landmarks):
    
    # be careful of the types: face_lanmarks is a list of mediapipe normalized lists.
    # i.e each element of this list holds a list of dictionaries each representing a landmark (dict name)
    # with its 3d coordinates as values (keys: x, y, z)

    if face_landmarks:
        for face in face_landmarks:
            # drawing_utils.draw_landmarks(image, face)
            landmarks = face.landmark  # returns a list of landmarks (containing coordinates)
        
            # looping only over landmarks corresponding to the left eye
            for id, landmark in enumerate(landmarks[473:478]):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                cv2.circle(image, (x, y), 2, (0, 0, 255))

                if id == 0:
                    pyautogui.moveTo(int(screen_w / frame_width * x), int(screen_h / frame_height * y))

            left_eye = [landmarks[145] , landmarks[159]]
            for point in left_eye:
                x = int(point.x * frame_width)
                y = int(point.y * frame_height)
                cv2.circle(image, (x, y), 2, (0, 255, 255))

            if abs(left_eye[0].y - left_eye[1].y) <=0.01:
                pyautogui.click()
                print("MOUSE CLICKED", left_eye[0].y - left_eye[1].y)
                # pyautogui.sleep(2)
                
while True:
    _, image = webcam.read()
    image = cv2.flip(image, 1)
    frame_height, frame_width, _ = image.shape

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processed_image = face_mesh_landmarks.process(rgb_image)
    all_face_landmarks = processed_image.multi_face_landmarks

    processFaceLandmarks(all_face_landmarks)
    cv2.imshow("image", image)

    key = cv2.waitKey(10)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
