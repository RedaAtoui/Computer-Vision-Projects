import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_specs = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    _, image = webcam.read()

    start = time.time()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True

    # convert back to bgr to be able to draw on the images
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # image dimensions to be used for scaling
    height, width, depth = image.shape

    face_3d =[]
    face_2d =[]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for id, landmark in enumerate(face_landmarks.landmark):
                if id == 1 or id == 33 or id == 61 or id == 199 or id == 263 or id == 291:
                    if id == 1:
                        nose_2d = (landmark.x * width, landmark.y * height)
                        nose_3d = (landmark.y * width, landmark.y * height, landmark.z * 3000)
                    
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    
                    face_2d.append([x, y])
                    face_3d.append([x, y, landmark.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)
            
            # tune this parameter based on camera calibration
            focal_length = 1 * width
            cam_matrix = np.array([[focal_length, 0, height / 2],
                                   [0, focal_length, width / 2],
                                   [0, 0, 1]
                                   ])

            # Distance matrix that is used for distortion
            distance_matrix = np.zeros((4, 1), dtype=np.float64)

            _, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, distance_matrix)
            rmat, _ = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text = " Looking Left"
            elif y > 10:
                text = " Looking Right"
            elif x < -10:
                text = " Looking Down"
            elif x > 10:
                text = " Looking Up"
            else:
                text = " Looking Forward"
            
            nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, distance_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, "x: " + str(np.round(x, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "y: " + str(np.round(y, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "z: " + str(np.round(z, 2)), (500, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime

            cv2.putText(image, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_specs, drawing_specs)

            cv2.imshow('HEAD POSE ESTIMATION', image)
            
            key = cv2.waitKey(10)
            if key == 27:
                break

webcam.release()
cv2.destroyAllWindows()