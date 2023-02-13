from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os, random

face_classifier = cv2.CascadeClassifier(r'E:\Deep Learning\Emotion\haarcascade_frontalface_default.xml')
classifier =load_model(r'E:\Deep Learning\Emotion\model.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)



while True:
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)


            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            
            if label == 'Happy':
                filename = random.choice(os.listdir("Happy/"))
                vid = cv2.VideoCapture("Happy/"+filename)
                
                while(vid.isOpened()):
                    ret, frame1 = vid.read()
                    if ret == True:
                        
                        cv2.imshow('Frame',frame1)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        else:
                            break
                    
            elif label == 'Sad':
                filename = random.choice(os.listdir("Sad/"))
                vid = cv2.VideoCapture("Sad/"+filename)
                
                while(vid.isOpened()):
                    ret, frame1 = vid.read()
                    if ret == True:
                        
                        cv2.imshow('Frame',frame1)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        else:
                            break
                        
            elif label == 'Neutral':
                filename = random.choice(os.listdir("Neutral/"))
                vid = cv2.VideoCapture("Neutral/"+filename)
                
                while(vid.isOpened()):
                    ret, frame1 = vid.read()
                    if ret == True:
                        
                        cv2.imshow('Frame',frame1)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        else:
                            break
                        
            elif label == 'Surprise':
                filename = random.choice(os.listdir("Surprise/"))
                vid = cv2.VideoCapture("Surprise/"+filename)
                
                while(vid.isOpened()):
                    ret, frame1 = vid.read()
                    if ret == True:
                        
                        cv2.imshow('Frame',frame1)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        else:
                            break
                        
                        
            elif label == 'Angry':
                filename = random.choice(os.listdir("Angry/"))
                vid = cv2.VideoCapture("Angry/"+filename)
                
                while(vid.isOpened()):
                    ret, frame1 = vid.read()
                    if ret == True:
                        
                        cv2.imshow('Frame',frame1)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                        else:
                            break
                        
                vid.release()
                cv2.destroyAllWindows()
            
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


























