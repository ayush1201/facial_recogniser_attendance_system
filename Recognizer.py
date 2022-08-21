import cv2
import numpy
import pyttsx3
import datetime


engineio = pyttsx3.init() #initializing engine

def markattendence(name):
                with open('attendence.csv','r+') as f:
                    mydatalist=f.readlines()
                    nameList = []
                    for line in mydatalist:
                        entry = line.split(',')
                        nameList.append(entry[0])
                    if name not in nameList:
                        now = datetime.datetime.now()
                        dtString = now.strftime('%H:%M:%S')
                        f.writelines(f'\n{name},{dtString}')

read = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

rec = cv2.face.LBPHFaceRecognizer_create() 
rec.read("recognizer/TraningData.yml") #loading the training data
font = (cv2.FONT_HERSHEY_COMPLEX_SMALL)

f = open("datatext.txt","r")
user = {}
for x  in f:
        y,z = x.split(" ")
        user[y] = z.replace("\n","")

while(1):
    status,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = read.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        id,conf = rec.predict(gray[y:y+h,x:x+w]) #returns id and confidence level
        print(id)
        if(conf>75):   
            name = "Unknown"
            
        else:
            name = user[str(id)]
            markattendence(name)

        engineio.say(name)
        engineio.runAndWait()
        
        cv2.putText(img,str(name),(x,y+h),font,255,(0,255,0))
        
    cv2.imshow('FaceDetect',img)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break

f.close()
cap.release()
cv2.destroyAllWindows()
    