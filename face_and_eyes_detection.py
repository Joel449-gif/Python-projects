import cv2
import numpy as np

cap = cv2.VideoCapture(0)   #takes parameter as the id of the camera through which the object has to capture. Here 0 stands for the first camera or primary/main camera, if your system has a lot of cameras, you can specify their nuber as 1,2,3,etc, depends on which the number of cameras you want to turn on.
# cap is variable that stores all the frames that have been captured. 

#path for face and eye cascade, path where it is storedspecific classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # takes 2 compulsory parameter 1)Source 2)color space conversion code
    faces = face_cascade.detectMultiScale(gray,1.3,5)   #detectMultiScale() - takes 3 parameter 1)image/source 2)scale factor -> specifies how much the image size is reduced at each image scale. 3)min Neighbors -> parameter specifying how many neighbors each candidate rectangle should have to retain it.
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)    #Draws the rectangle over the image/ video capture  over the source and takes parameters as 1)source 2)starting point co-ordinates 3)ending point co-ordinates 4)color value(RGB) 5)thickness in px(-1 to fill)

        roi_gray = gray[y:y+w,x:x+w]
        roi_color = frame[y:y+h,x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0))
    
    cv2.imshow('Frame', frame)
    
    if(cv2.waitKey(1) == ord('e')): # ord() - desired character that must be press to escape
        break   #here loop is used to close the camera , Hence waitkey() is compared with the ord()
cap.release()   # camera is released so that other applications can use if it's required.
cv2.destroyAllWindows()
