#!/usr/bin/env python
# coding: utf-8

# In[40]:


import cv2
import numpy as np 
import dlib


# In[47]:


cap = cv2.VideoCapture("http://192.168.0.102:8080/video") # webcam url
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

_, frame = cap.read()
rows, cols, _ = frame.shape    # (height,width,channel)

smoke = cv2.imread("smoke.png")

h, w, _ = smoke.shape
for i in range(h):
    for j in range(w):
        x = smoke[i,j]
        flag=0
        for h in x:
            if h<=200:
                flag=1
                break
        if flag==0:
            smoke[i,j]=(0,0,0)
            
smoke_mask=np.zeros((rows, cols), dtype='uint8')

while True:
    _, frame = cap.read()   #read the frame

    smoke_mask.fill(0)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   #converting in grayscale
    faces = detector(gray)


    for face in faces:    
        a1 = face.left()
        b1 = face.top()
        a2 = face.right()
        b2 = face.bottom()
        faceHeight = b2 - b1
        faceWidth = a2- a1
        landmarks = predictor(gray, face) #68 points landmark
        up= (landmarks.part(62).x ,landmarks.part(62).y) 
        down= (landmarks.part(66).x ,landmarks.part(66).y) 
        smoke_left =(landmarks.part(60).x ,landmarks.part(60).y) 
        smoke_right =(landmarks.part(61).x ,landmarks.part(61).y) 
        
        smoke_width = smoke_right[0] - smoke_left[0]
        smoke_width =  4 * smoke_width
        smoke_height = 3 * smoke_width
        
        mid = ((smoke_left[0] + smoke_right[0])/2,(smoke_left[1] + smoke_right[1])/2)
        top = (int(mid[0] - smoke_width / 7),int(mid[1] - smoke_height / 7))
        #Bottom = (int(mid[0] + smoke_width / 2),int(mid[1] + smoke_height / 2))

        
        if (down[1]-up[1])/faceHeight>0.05:
            
            s_=cv2.resize(smoke , (int(smoke_width) , int(smoke_height)))
            smoke_gray=cv2.cvtColor(s_,cv2.COLOR_BGR2GRAY)
            smoke_area = frame[top[1] : top[1] + smoke_height, top[0] : top[0] + smoke_width]  
            _, smoke_mask = cv2.threshold(smoke_gray, 25, 255, cv2.THRESH_BINARY_INV)
            smoke_area_no_t = cv2.bitwise_and(smoke_area, smoke_area, mask = smoke_mask)
            final_s = cv2.add(smoke_area_no_t, s_)
            frame[top[1] : top[1] + smoke_height, top[0] : top[0] + smoke_width]  = final_s
                  



    
    cv2.imshow("Smoke Filter Frame",frame)



    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        cap.release()
        cv2.destroyAllWindows()
        break


# In[ ]:




