import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
data_path='D:/Projects/Faces/'
onlyfiles=[f for f in listdir(data_path)if isfile(join(data_path,f))]


Training_Data,labels =[], []

for i,  files in enumerate(onlyfiles):
    image_path=data_path+onlyfiles[i]
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images,dtype=np.uint8))
    labels.append(i)

labels=np.asarray(labels,dtype=np.int32)

model=cv2.face.LBPHFaceRecognizer_create()  ## Linear Binary Phase Histogram(LBPH)
model.train(np.asarray(Training_Data),np.asarray(labels))

print("!!! Model Training Complete, Ready to Launch !!! ")

face_classifier= cv2.CascadeClassifier('C:/python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def face_detector(image,size=0.5):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return image,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,255),2)
        roi=image[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))

    return  image,roi

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()

    image, face=face_detector(frame)

    try:

        face=cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
        result=model.predict(face)
        if result[1]<500:
            conf=int(100*(1-(result[1])/300))
            dispaly_string=str(conf)+'% Face Matches'
        cv2.putText(image,dispaly_string,(100,200),cv2.FONT_HERSHEY_DUPLEX,1,(250,120,255),2)
        if conf>70:
            cv2.putText(image,"Unloking The System", (230, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Image Cropper",image)
        else:
            cv2.putText(image, "Invalid User, System Can not be unlocked", (230, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Image Cropper", image)




    except:

        cv2.putText(image, "No Face Detected", (230, 450), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Image Cropper", image)
        pass


    if cv2.waitKey(1)==13:
        break

cap.release()

cv2.destroyAllWindows()





