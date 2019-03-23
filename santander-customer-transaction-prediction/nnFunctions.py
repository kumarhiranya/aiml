# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:46:10 2019

@author: Tensorbook 4
"""
import keras as K
from keras import layers
from keras.models import Sequential
from dataFunctions import *
import numpy as np
import pandas as pd
import keras
import numpy as np
import sklearn.metrics as sklm


class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        self.confusion.append(sklm.confusion_matrix(targ, predict))
        self.precision.append(sklm.precision_score(targ, predict))
        self.recall.append(sklm.recall_score(targ, predict))
        self.f1s.append(sklm.f1_score(targ, predict))
        self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return

def getLayer(n, conv=False):
    '''
    Helper function for createModel function. Returns dropout layer ot dense layer depending on the input.
    If n is int, returns Dense layer of n nodes
    If n is float, returns Dropout layer with n as the dropout factor.
    '''
    if type(n)==float:
        return layers.Dropout(n, noise_shape=None, seed=None)
    elif type(n)==int:
        return layers.Dense(n, activation = "relu")
    
def createModel(nInputs, nOutputs=1, hiddenLayers = None, binary = True):
    '''
    Returns a keras ANN model with the specified parameters.
    nInputs: Number of nodes in the input layer.
    nOutputs: Number of nodes in the output layer.
    hiddenLayers: a list of length n, where n is the total number of hidden layers (including dropout layers). Each element in the list either the number of nodes in a Dense layer or the dropout factor of a Dropout layer depending on weather n is int or float.
    binary: Type of classification. Decides the loss function used.
    
    Returns a Keras Sequential model.
    '''
    model = Sequential()
    if not hiddenLayers:
        # Input - Layer
        model.add(layers.Dense(500, activation = "relu", input_shape=(nInputs, )))

        # Hidden - Layers
        model.add(layers.Dropout(0.4, noise_shape=None, seed=None))
        model.add(layers.Dense(50, activation = "relu"))
    else:
        # Input - Layer
        model.add(layers.Dense(hiddenLayers[0], activation = "relu", input_shape=(nInputs, )))
        
        #Hidden layers
        for l in hiddenLayers[1:]:
            model.add(getLayer(l))
    
    # Output- Layer
    model.add(layers.Dense(nOutputs, activation = "sigmoid"))
#     model.summary()
    if binary:
        model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    else:
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
    return model

def cnn1d(nInputs, nOutputs=1, hiddenLayers = None, batchSize = 50, binary = True):
    from keras.models import Sequential
    from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense
    import keras
    """
    Needs work.
    """
    model = Sequential()
    if hiddenLayers==None:
        model.add(Conv1D(filters=20, kernel_size=100, activation='relu', 
                         input_shape=(nInputs, 1)))
        print(model.input_shape)
        print("conv1:", model.output_shape)
        model.add(MaxPooling1D(pool_size=10, strides=4))
        print("MaxPool1:", model.output_shape)
#         model.add(Conv1D(kernel_size = (200), filters = 20, input_shape=nInputs, activation='relu'))
#         print(model.input_shape)
#         print(model.output_shape)
#         model.add(MaxPooling1D(pool_size = (20), strides=(7)))
#         print(model.output_shape)
        model.add(keras.layers.core.Reshape([20,-1,1]))
        print("Reshape1:", model.output_shape)    
        model.add(Conv2D(kernel_size = (20,30), filters = 400, activation='relu'))
        print("conv2_2D:", model.output_shape)
        model.add(MaxPooling2D(pool_size = (1,10), strides=(1,2)))
        print("MaxPool2_2D:",model.output_shape)
        model.add(Flatten())
        print("Flat1:", model.output_shape)
        model.add(Dense (500, activation='relu'))
        print("Dense1:", model.output_shape)
        model.add(Dense (100, activation='relu'))
        print("Dense2:", model.output_shape)
        model.add(Dense(nOutputs, activation = 'softmax',activity_regularizer=keras.regularizers.l2()  ))
        print("Dense3:", model.output_shape)
        
    model.compile( loss='binary_crossentropy', optimizer=keras.optimizers.SGD(), metrics=[keras.metrics.categorical_accuracy])
    return model 

def genLayers(l1, l2, l3):
    '''
    Helper generator function to search for optimal number of nodes and dropout factor for a 3 layered NN.
    Generates all the possible combinations of elements in l1, l2 and l3.
    '''
    for x in l1:
        for y in l2:
            for z in l3:
                yield [x,y,z]

def gridSearchNN(newTrainX, newTrainY, newTestX, newTestY, l1 = [100,200,400,600],
                  l2 = [0.2,0.3,0.5], l3 = [10,20,40,60], epochs = 20,
                  batchSize = 100, filename = '', verbose = 0, binary = False):
    '''
    Builds a model for all the possible layer configurations for the mentioned parameters in l1, l2 and l3.
    Returns a dataframe containing parameters and test accuracies and F1 scores for each of the generated models and for each of the categories.
    filename : Name of the file to write the results into, if empty no file is written.
    binary: Type of classification.
    epochs and batchSize : parameters for the .fit method of the model.
    '''
    if binary:
         results = ['Layer1 Layer2 Layer3 Epochs BatchSize'.split()+['F1 Score', 'Accuracy']]        
         nOutputs = 1
    else:
         results = ['Layer1 Layer2 Layer3 Epochs BatchSize'.split()+['F1-'+ x for x in newTrainY.columns]+['F1 Average', 'Accuracy Avg']]
         nOutputs = newTrainY.shape[1]
    cnt = 1
    fits = len(l1)*len(l2)*len(l3)
    if verbose==1:
         print("Total fits = ", fits)
         print(results)
     
     
    for hiddenLayers in genLayers(l1=l1,l2=l2,l3=l3):
        model2 = createModel(newTrainX.shape[1], nOutputs=nOutputs, hiddenLayers=hiddenLayers,
                        binary=binary)
        
        results2 = model2.fit(newTrainX, newTrainY, epochs = epochs, 
                            batch_size = batchSize, validation_data = (newTestX, newTestY), verbose=0)
        
        if not binary:
            newTestYPred = pd.get_dummies(model2.predict_classes(newTestX))
            newTestYPred.columns = newTestY.columns
        else:
            newTestYPred = model2.predict_classes(newTestX)
            print(newTestYPred.shape)
        confusionTest1 = confusionMatMulti(newTestY, newTestYPred, binary=binary, className='OrderStatus') 
        if(verbose==1):
            print(hiddenLayers, epochs, batchSize, list(confusionTest1['F1 Score']))
        if not binary:
            temp = hiddenLayers + [epochs, batchSize] + list(confusionTest1['F1 Score']) + [np.average(confusionTest1['F1 Score'][:-1]), np.average(confusionTest1['Accuracy'][:-1])]
        else:
            temp = hiddenLayers + [epochs, batchSize] + [confusionTest1['F1 Score'][0], confusionTest1['Accuracy']]
        results.append(temp)
        print(cnt/fits*100,"%")
        cnt+=1
         
 #         print(temp)
 
    resultsDf = pd.DataFrame(results)
    if filename!='':
        resultsDf.to_csv('Results/'+filename+'.csv', index=False)
    return resultsDf

def evaluateModel(model, xTest, yTest, modelType=None):
    '''Takes in a fitted models and gives out metrics for test data provided in xTest and yTest'''
    if modelType!=None:
        yTestPred = model.predict_classes(xTest)
    else:
        yTestPred = model.predict(xTest)
    if len(yTest.shape)>1:
        yTestPred = pd.get_dummies(yTestPred)
        yTestPred.columns = yTest.columns
        cm = confusionMatMulti(yTest, yTestPred, binary=False)
    else:
        cm = confusionMatMulti(yTest, yTestPred, binary=True)
    return cm

def plotModelHistory(results):
    from matplotlib import pyplot as plt
    # list all data in history
    keys = list(results.history.keys())
    # summarize history for accuracy
    plt.plot(results.keys[1])
    plt.plot(results.keys[3])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xticks([i for i in range(0, len(results.history[keys[0]]), 10)])
    plt.legend(['train', 'test'], loc='best')
    plt.xticks([i for i in range(len(results.history['acc']), len(results.history['acc'])//10)])
    plt.show()
    # summarize history for loss
    plt.plot(results.history[keys[2]])
    plt.plot(results.history[keys[0]])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xticks([i for i in range(0, len(results.history['acc']), 10)])
    plt.legend(['train', 'test'], loc='best')
    plt.show()
    
def saveKerasModel(model, filename):
    '''
    Saves Keras ANN model and its weights. Model is saved in a .yaml file while the weights are stored in a .h5 file.
    '''
    model.save("Models/"+filename+".hd5")
    
def precision_threshold(threshold=0.5):
    def precision(y_true, y_pred):
        """Precision metric.
        Computes the precision over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio
    return precision

def recall_threshold(threshold = 0.5):
    def recall(y_true, y_pred):
        """Recall metric.
        Computes the recall over the whole batch using threshold_value.
        """
        threshold_value = threshold
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio
    return recall

def nnCrossValidator(x, y, hiddenLayers, epochs=15, batchSize=200, k=3, binary = False, filename = ''):
    crossValResult = []
#     columns = ['F1 Test Split '+str(i+1) for i in range(k)]+
#     ['Average F1 Train','Average F1 Test','Average Accuracy Train', 'Average Accuracy Test']
    columns = ['Split', 'Train F1', 'Test F1', 'Train Acc', 'Test Acc']
    cnt=1
    for xTrain, xTest, yTrain, yTest in kFold(x, y, k=k, binary = binary):
#         print(yTrain.shape, yTrain.head())
        tempModel = createModel(x.shape[1], hiddenLayers=hiddenLayers)
        tempModel.fit(xTrain, yTrain, epochs = epochs, batch_size = batchSize, verbose=0)
        if binary:
            yPredTrain = tempModel.predict_classes(xTrain)
            yPredTest = tempModel.predict_classes(xTest)
        else:
            yPredTrain = pd.get_dummies(tempModel.predict_classes(xTrain))
            yPredTrain.columns = yTrain.columns
            
            yPredTest = pd.get_dummies(tempModel.predict_classes(xTest))
            yPredTest.columns = yTest.columns
        
        
        trainRes = confusionMatMulti(yTrain, yPredTrain, binary = binary)
        testRes = confusionMatMulti(yTest, yPredTest, binary = binary)
        temp = ['Split '+str(cnt), list(trainRes['F1 Score']), list(testRes['F1 Score']),
                              list(trainRes['Accuracy']), list(testRes['Accuracy'])]
        crossValResult.append(temp)
#         print(temp)
        cnt+=1
    crossValResult = pd.DataFrame(crossValResult)
    crossValResult.columns = columns
    if filename != '':
        crossValResult.to_csv("Results/"+filename+'.csv')
    return crossValResult