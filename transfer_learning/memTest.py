# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 22:02:04 2019

@author: khira
"""

#import numpy as np
#from keras.datasets import cifar10
#from matplotlib import pyplot as plt
#from sys import getsizeof
from memory_profiler import profile
#%matplotlib inline

#Load the dataset:
@profile(precision=4)
def loadData():
    from keras.datasets import cifar10
    data = cifar10.load_data()
    (X_train, y_train),(X_test, y_test) = data
#    del data[1]
#    print(len(data))
    return X_train, y_train, X_test, y_test

@profile(precision=4)
def createModel():
    #Importing the necessary libraries 
    from keras.models import Sequential
    from keras.layers import Dense, Conv2D, MaxPooling2D
    from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
    
    #Building up a Sequential model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu',input_shape = X_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(GlobalAveragePooling2D())
    model.add(Dense(10, activation='softmax'))
#    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

@profile(precision=4)
def fitAndScoreModel(X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0, callbacks=[checkpointer], validation_split=0.2, shuffle=True)

    #Evaluate the model on the test data
    score = model.evaluate(X_test, y_test)
    
    #Accuracy on test data
    print('Accuracy on the Test Images: ', score[1])
    return model, score

@profile
def loadResnet():
    #Importing the ResNet50 model
    from keras.applications.resnet50 import ResNet50, preprocess_input
    from skimage.transform import resize
    
    #Loading the ResNet50 model with pre-trained ImageNet weights
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    return model

@profile
def resizeImages(images):
    from skimage.transform import resize
    import numpy as np
    
    #Reshaping the training data
    X_train_new = np.array([resize(X_train[i], (200, 200, 3)) for i in range(0, len(X_train))]).astype('float32')
    return X_train_new
    

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = loadData()
    num_classes = 10

    from keras.utils import np_utils
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)
    
#    model = createModel()
    
#    from keras.callbacks import ModelCheckpoint
#    checkpointer = ModelCheckpoint(filepath='scratchmodel.best.hdf5', verbose=1,save_best_only=True)
    
#    model, score = fitAndScoreModel(X_train, y_train, X_test, y_test)
    
    X_train_new = resizeImages(X_train)
    
    resnet = loadResnet()
    

    
# X_train = X_train[:5000]
# y_train = y_train[:5000]
# X_test = X_test[:2500]
# y_test = y_test[:2500]
    
#@profile(precision=4)
#def my_func():
#    a = [1] * (10 ** 6)
#    b = [2] * (2 * 10 ** 7)
#    del b
#    return a
#
#if __name__ == '__main__':
#    my_func()