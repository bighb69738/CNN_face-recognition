import cv2
import os
import numpy as np
import keras

def loadImages():
    imageList=[]
    labelList=[]

    rootdir="/home/user/vic 小程式/CNN_classify/book_photo/my_face"
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (200, 200))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(0)#類別0

    rootdir="/home/user/vic 小程式/CNN_classify/book_photo/BW_faces"
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (200, 200))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(1)#類別1


    rootdir="/home/user/vic 小程式/CNN_classify/book_photo/CYM_faces"
    list =os.listdir(rootdir)
    for item in list:
        path=os.path.join(rootdir,item)
        if(os.path.isfile(path)):
            f=cv2.imread(path)
            f=cv2.resize(f, (200, 200))#resize到網絡input的shape
            imageList.append(f)
            labelList.append(2)#類別1

    return np.asarray(imageList), keras.utils.to_categorical(labelList, 3)
