import cv2
import os
import numpy as np
import keras
from photo import loadImages
from CNN_model import Net_model
from keras.preprocessing.image import ImageDataGenerator


def loadImages():
    imageList=[]

    rootdir="/home/user/vic 小程式/CNN_classify/book_photo/validate"
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (200, 200))
            imageList.append(f)

    return np.asarray(imageList)


def convert2label(vector,x):
    string_array=[]
    yes=0
    no=0
    for v in vector:
        if v==0:
            string_array.append('HH')
            cv2.imwrite('/home/user/vic 小程式/CNN_classify/book_photo/YES/img'+str(yes)+'.jpg', x[len(string_array)-1])
            yes+=1
        else:
            string_array.append('NOT HH')
            cv2.imwrite('/home/user/vic 小程式/CNN_classify/book_photo/NO/img'+str(no)+'.jpg', x[len(string_array)-1])
            no+=1
    return string_array


x=loadImages()
x=np.asarray(x)



model=Net_model(nb_classes=2, lr=0.0001)
model.load_weights("/home/user/vic 小程式/CNN_classify/trained_myface_model_weights.h5")

print(model.predict(x))
print(model.predict_classes(x))
y=convert2label(model.predict_classes(x),x)

print(y)

for i in range(len(x)):
    cv2.putText(x[i], y[i], (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    cv2.imshow('image'+str(i), x[i])


cv2.waitKey(-1)