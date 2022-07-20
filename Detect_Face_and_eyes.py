from ast import Break
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

#path for face and eye cascade, path where it is storedspecific classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # takes 2 compulsory parameter 1)Source 2)color space conversion code
    faces = face_cascade.detectMultiScale(gray,1.3,5)   #detectMultiScale() - takes 3 parameter 1)image/source 2)scale factor -> specifies how much the image size is reduced at each image scale. 3)min Neighbors -> parameter specifying how many neighbors each candidate rectangle should have to retain it.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)

        roi_gray = gray[y:y+w,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0))
    
    cv2.imshow('Frame', frame)
    
    if(cv2.waitKey(1) == ord('e')):
        break
cap.release()
cv2.destroyAllWindows()