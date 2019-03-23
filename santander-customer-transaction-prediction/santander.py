# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:15:48 2019

@author: khira
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from nnFunctions import *

def runNEval(model, x_train, y_train, x_test, y_test, returnResults=False):
    model.fit(x_over, y_over)
    trainPred = model.predict(x_train)
    testPred = model.predict(x_test)
    
    trainResult = classification_report(y_train, trainPred)
    testResult = classification_report(y_test, testPred)
    
    print('Train results:', accuracy_score(y_train, trainPred), f1_score(y_train, trainPred), roc_auc_score(y_train, trainPred))
    print('Test results', accuracy_score(y_test, testPred), f1_score(y_test, testPred), roc_auc_score(y_test, testPred))
    
    if returnResults:
        return trainResult, testResult, model

import tensorflow as tf
from keras import backend as K

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
    
data = pd.read_csv('train.csv')

testData = pd.read_csv('test.csv')
testLables = testData.iloc[:,0]
xTest = testData.iloc[:,1:]

#posData = data[data.target==1]
#negData = data[data.target==0]
#posStats = posData.describe().drop('target',axis=1).drop('count',axis=0)
#negStats = negData.describe().drop('target',axis=1).drop('count',axis=0)

x = data.iloc[:,2:]
y = data.iloc[:,1]
x_train, x_test, y_train, y_test = train_test_split(x,y)
smote = SMOTE(sampling_strategy=0.6, k_neighbors=5, n_jobs=-1)
x_over, y_over = smote.fit_resample(x_train,y_train)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model

nInputs = x.shape[1]
nOutputs = 1

model = Sequential()
model.add(Dense(500, activation='relu', input_shape=(nInputs,)))
model.add(Dropout(0.4))
model.add(Dense(500, activation='relu', input_shape=(nInputs,)))
model.add(Dropout(0.4))
model.add(Dense(10, activation='relu'))
model.add(Dense(nOutputs, activation='sigmoid'))
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = [auc])

batchSize=100
epochs = 200

#model = load_model('Models/ann0.hd5')
hist = model.fit(x_over, y_over, batch_size=batchSize, epochs=epochs, 
                 validation_split=0.1)

#saveModel(hist, 'hist_500_10_500e')
#saveKerasModel(model, 'ann_500_500_10_500e')

#model2 = load_model('Models/ann_500_500_10_500e.hd5')
#hist = loadModel('hist_500_10_500e')

predTrain = model.predict_classes(x_train)
predTest = model.predict_classes(x_test)

print(f1_score(y_train, predTrain), f1_score(y_test, predTest))

plotModelHistory(hist)

testPred = model.predict_classes(xTest).reshape((len(xTest),))
subDf = pd.concat([testLables, pd.Series(testPred)],axis=1, ignore_index=True)
subDf.columns = ['ID_code', 'target']
subDf.to_csv('submission3.csv', index=False)

#bestParams = {'bootstrap': True, 'class_weight': None, 'criterion': 'entropy', 'max_depth': 100, 
#              'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 
#              'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 
#              'min_weight_fraction_leaf': 0.0, 'n_estimators': 1000, 'n_jobs': 7, 
#              'oob_score': False, 'random_state': None, 'verbose': 0, 'warm_start': False}
#model_rf = RandomForestClassifier(**bestParams)
#
##model_rf = RandomForestClassifier(n_estimators=400, criterion='entropy', max_depth=100,
##                                  min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
##                                  max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
##                                  min_impurity_split=None, bootstrap=True, oob_score=False,
##                                  n_jobs=-1, random_state=None, verbose=0, warm_start=False,
##                                  class_weight=None)
#
#
#params = {'n_estimators':[200, 500, 800, 1000], 'criterion':['entropy','gini'], 'max_depth':[50, 100,150,200],
#                                  'min_samples_split':[2], 'min_samples_leaf':[1], 'min_weight_fraction_leaf':[0.0],
#                                  'max_features':['auto'], 'max_leaf_nodes':[None], 'min_impurity_decrease':[0.0],
#                                  'min_impurity_split':[None], 'bootstrap':[True], 'oob_score':[False],
#                                  'n_jobs':[1], 'random_state':[None], 'verbose':[0], 'warm_start':[False],
#                                  'class_weight':[None]}
#                                  
#trainResult, testResult, fitModel = runNEval(model_rf, x_over, y_over, x_test, y_test, returnResults=True)
#
#testPred = fitModel.predict(xTest)
#subDf = pd.concat([testLables, pd.Series(testPred)],axis=1, ignore_index=True)
#subDf.columns = ['ID_code', 'target']
#subDf.to_csv('submission.csv', index=False)


#gridSearchResults = GridSearchCV(RandomForestClassifier(), param_grid=params, n_jobs=7,
#                                 scoring=['accuracy'], verbose=1, refit=False)
#gridSearchResults.fit(x_over, y_over)
#model_rf.fit(x_over, y_over)
#trainPred = model_rf.predict(x_train)
#testPred = model_rf.predict(x_test)
#
#trainResult = classification_report(y_train, trainPred)
#testResult = classification_report(y_test, testPred)
#
#print('Train results:', accuracy_score(y_train, trainPred), f1_score(y_train, trainPred), roc_auc_score(y_train, trainPred))
#print('Test results', accuracy_score(y_test, testPred), f1_score(y_test, testPred), roc_auc_score(y_test, testPred))
