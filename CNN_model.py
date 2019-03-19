from keras.models import Sequential  
from keras.layers.core import Dense, Dropout, Activation, Flatten  
from keras.layers.convolutional import Convolution2D, MaxPooling2D  
from keras.optimizers import SGD  


def Net_model(nb_classes, lr=0.001,decay=1e-6,momentum=0.9):  
    model = Sequential()  
    model.add(Convolution2D(filters=10, kernel_size=(5,5),
                            padding='valid',  
                            input_shape=(200, 200, 3)))  
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
  
    model.add(Convolution2D(filters=20, kernel_size=(10,10)))
    model.add(Activation('tanh'))  
    model.add(MaxPooling2D(pool_size=(2, 2)))  
    model.add(Dropout(0.25))  
  
    model.add(Flatten())  
    model.add(Dense(1000))
    model.add(Activation('tanh'))  
    model.add(Dropout(0.5))  
    model.add(Dense(nb_classes))  
    model.add(Activation('softmax'))  
  
    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd)  
      
    return model  
