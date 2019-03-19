nb_classes = 3  
nb_epoch = 30
nb_step = 6
batch_size = 3

from photo import loadImages
from CNN_model import Net_model
from keras.preprocessing.image import ImageDataGenerator
x,y=loadImages()


dataGenerator=ImageDataGenerator()
dataGenerator.fit(x)
data_generator=dataGenerator.flow(x, y, batch_size, True)#generator函數，用來生成批處理數據（從loadImages中）

model=Net_model(nb_classes=nb_classes, lr=0.0001) #加載網絡模型

history=model.fit_generator(data_generator, epochs=nb_epoch, steps_per_epoch=nb_step, shuffle=True)#訓練網絡，並且返回每次epoch的損失value

model.save_weights("/home/user/vic 小程式/CNN_classify/trained_face_model_20180910_weights.h5")#保存權重
print("DONE, model saved in path-->/home/user/vic 小程式/CNN_classify/trained_face_model_20180910_weights.h5")