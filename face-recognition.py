import cv2


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
print('hie there')
while True:
    _, img = webcam.read()
    print('ho there')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.7, 4)
    print("FACES", faces)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)

    cv2.imshow('FACE', img)
    cv2.imshow('gray', gray)
    
    key = cv2.waitKey(delay=10)
    if key ==  27:
        break
webcam.release()
cv2.destroyAllWindows() 
