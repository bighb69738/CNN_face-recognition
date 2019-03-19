import cv2
import os
import numpy as np
import keras
import sys

    #rootdir="/home/user/Downloads/CNN_classify/book_photo/book"
    #path = "/var/www/html/"
dirs = os.listdir("/home/user/Downloads/CNN_classify/book_photo/book")
path = "/home/user/Downloads/CNN_classify/book_photo/book/book8.jpg"
if(os.path.isfile(path)):
    print("Y")
else:
    print("N")

for  test  in dirs:
    print(test)