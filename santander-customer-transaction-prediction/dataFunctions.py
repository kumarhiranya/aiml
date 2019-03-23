# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 16:09:07 2019

@author: Tensorbook 4
"""

import numpy as np 
import pandas as pd
import re
# from merger import merger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from nltk import sent_tokenize, word_tokenize
      
import pickle
#from email_preprocessing import email_preprocessing
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, classification_report

#from hypopt import GridSearch

# Show all outputs in a cell instead of only the last one
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Supresses warnings for assignment of individual series in a dataframe
#pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
#%matplotlib inline


def createCsv(x,y,pred,filename= '', extraCols = []):
    '''
    Creates and returns a DataFrame with columns x (Email), y (OM Label), pred (Model predictions).
    Writes the DataFrame to filename if filename is passed.
    '''
    if type(extraCols) != list:
        testDf = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True),
                            pd.Series(pred), extraCols.reset_index(drop=True)], axis=1)
        testDf.columns = ["Email", "OM Label", "Prediction"] + list(extraCols.columns)
    else:
        testDf = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True),
                            pd.Series(pred)], axis=1)
        testDf.columns = ["Email", "OM Label", "Prediction"]

    if filename!='':
        testDf.to_csv("Results/"+filename+".csv", index=False)
    return testDf

def prepareCascadeData(x, df_y, yPred, catColIndices):
    '''
    Parameters:
        x - input data
        df_y - output true labels in a DataFrame with each column conatining binary values for the corresponding category
        yPred - Binary prediction labels for the class to be filtered upon, in (n_samples, 1) shape
        catColIndices - Columns indices corresponding to the categories to be extracted from df_y
    Returns:
        newX - input data
        newY - one hot values with the selected categories + 1 category of Other
    '''
    y_multi = df_y.iloc[:,catColIndices]
    other = pd.Series([int(1-x.sum()) for x in y_multi.values])
    if type(yPred)==pd.Series:
        newDf = pd.concat([pd.DataFrame(x), yPred,
                            y_multi.reset_index(drop=True), other], axis=1, ignore_index=True)
    else:
        newDf = pd.concat([pd.DataFrame(x), pd.Series(yPred[:,0]),
                            y_multi.reset_index(drop=True), other], axis=1, ignore_index=True)
    newDf.columns = [x for x in range(x.shape[1])]+['Pred']+list(y_multi.columns)+['Other']

    filteredDf = newDf[newDf['Pred']==0]
    newX = filteredDf.iloc[:, :x.shape[1]]
    newY = filteredDf.iloc[:, x.shape[1]+1:]
    return newX, newY


def getIndices(categoryNames, df_y):
    indList = []
    for cat in categoryNames:
        indList.append(df_y.columns.get_loc(cat))
    return indList

def confusionMatMulti(yTrue, yPred, binary=False, className = 'Class1'):
    '''
    Parameters:
    yTrue - Pandas DataFrame containing True values and correct category names as column names
    yPred - Pandas DataFrame containing Predictions
    
    Returns:
    Dictionary containing confusion matrix for each column in yTrue
    '''
    cm = []
    if not binary:
        categories = list(yTrue.columns)
        

    #     'tn_test':confusionMat_test[0], 'fp_test':confusionMat_test[1],
    #             'fn_test':confusionMat_test[2], 'tp_test':confusionMat_test[3]
        for i in range(yTrue.shape[1]):
            cm.append([categories[i]]+list(confusion_matrix(yTrue.iloc[:,i], yPred.iloc[:,i]).ravel())+
                     [f1_score(yTrue.iloc[:,i], yPred.iloc[:,i]), accuracy_score(yTrue.iloc[:,i], yPred.iloc[:,i])])
    else:
        cm.append([className]+list(confusion_matrix(yTrue, yPred).ravel())+
                     [f1_score(yTrue, yPred), accuracy_score(yTrue, yPred)])
    
    cmDf = pd.DataFrame(cm)
    cmDf.columns = ['Category', 'True Negative', 'False Positive', 'False Negetive', 'True Positive',
                   'F1 Score', 'Accuracy']
    return cmDf

def kFold(xAll, yAll, k = 3, binary = False):
    '''
    A generator function that takes in training data (vectors) and gives out k pairs of 
    training and test data.
    xAll - Training vectors (numpy array) of shape (n_samples, n_features)
    yAll - Single column (numpy array) containing the class lables
    k - integer denoting the number of folds to split in, defaults to 3
    '''
    splits = StratifiedKFold(n_splits = k, shuffle=True)
    for train_index, test_index in splits.split(xAll, yAll):
        X_train, X_test = xAll[train_index,:], xAll[test_index,:]
        y_train, y_test = yAll[train_index], yAll[test_index]
        if not binary:
            y_train = pd.get_dummies(y_train)
            y_test = pd.get_dummies(y_test)
        yield X_train, X_test, y_train, y_test
        
def crossValidator(model, x, y, k=3, binary = False, modelType = None, filename = ''):
    crossValResult = []
#     columns = ['F1 Test Split '+str(i+1) for i in range(k)]+
#     ['Average F1 Train','Average F1 Test','Average Accuracy Train', 'Average Accuracy Test']
    columns = ['Split', 'Train F1', 'Test F1', 'Train Acc', 'Test Acc']
    epochs = 20
    batchSize = 100
    cnt=1
    tempModel = model
    for xTrain, xTest, yTrain, yTest in kFold(x, y, k=k, binary = binary):
#         print(yTrain.shape, yTrain.head())
        tempModel.reset_states()
        if modelType != None:
            tempModel.fit(xTrain, yTrain, epochs = epochs, batch_size = batchSize, verbose=0)
            if binary:
                yPredTrain = tempModel.predict_classes(xTrain)
                yPredTest = tempModel.predict_classes(xTest)
            else:
                yPredTrain = pd.get_dummies(tempModel.predict_classes(xTrain))
                yPredTrain.columns = yTrain.columns
                
                yPredTest = pd.get_dummies(tempModel.predict_classes(xTest))
                yPredTest.columns = yTest.columns
            
        else:
            tempModel.fit(xTrain, yTrain)
            yPredTrain = tempModel.predict(xTrain)
            yPredTest = tempModel.predict(xTest)
        
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

def createTfidfSvd(textSeries, tfidfMaxFeatures = 15000, svdNComp = 1200, writeModel=False, tfidfFilename = 'tfidf_model_all.sav', svdFilename = 'svd_model_all.sav'):
     tfidf = TfidfVectorizer(max_features = tfidfMaxFeatures, lowercase=True, analyzer='word',ngram_range=(1,2))
     feature_vector = tfidf.fit_transform(textSeries)
     if writeModel:
         pickle.dump(tfidf, open(tfidfFilename, 'wb'))
    
     svd = TruncatedSVD(n_components=svdNComp, n_iter=7, random_state=0)
     reduced_vector = svd.fit_transform(feature_vector)
     if writeModel:
         pickle.dump(svd, open(svdFilename, 'wb'))
     
     return {'tfidf':tfidf, 'svd':svd, 'xVector':reduced_vector}
     
def saveModel(model, filename):
    '''
    Saves model with the passed filename in "Models" folder.
    '''
    pickle.dump(model, open('Models/'+filename, 'wb'))

def loadModel(filename):
    '''
    Loads model named filename from the Models folder.
    '''
    return pickle.load(open('Models/'+filename, 'rb'))

def tokenizeText(text, stop_words=[], tokenizeSent = False):
    '''
    Parameters:
        text: text
        stop_words: words to be not included
        tokenizeSent: If True, Seperates sentences in the text as seperate lists, returns a list of lists of tokens
    '''
    tokens = []
    for sent in sent_tokenize(text):
        if tokenizeSent:
            tokens.append([word for word in word_tokenize(sent) if word.isalpha() and word not in stop_words])
        else:
            tokens += [word for word in word_tokenize(sent) if word.isalpha() and word not in stop_words]
    return tokens

def getWordStrFraction(word,string):
    return len(word)*len(re.findall(word, string))/len(string)

def findOrderNumbers(textList):
    orderPatterns = {
        'NSPE' : 'nspe[0-9]{10}',
        'IQ' : 'iqbln[0-9][a-z][0-9]',
        'COMS IR' : '(?<=\s|\||:|,)ir\w{6}[(?<=\s)|\.|,|:|;]',
        'PQ' : '19[0-9]{7}',
        'OPRO' : '(?<=\s|\||:|,)9[0-9]{6}',
        'NCAP' : 'n\w{8},\s?n\w{8}',
        'ESP' : 'n[0-9]{7},\s?[0-9]{8}',
        'COMS' : 'oq\w{6},\s?cq\w{6}',
        }
        
    if type(textList)==list:
        textList = pd.Series(textList)
    
    if type(textList) == pd.core.series.Series:
        out = pd.DataFrame()
        out['Emails'] = textList
        textList = textList.str.lower()
        for pattern in orderPatterns.keys():
#            textList = textList.str.replace(orderPatterns[pattern], pattern)
            out[pattern] = textList.str.findall(orderPatterns[pattern], pattern)
    
    elif type(textList) == str:
        out = {}
        for pattern in orderPatterns.keys():
            out[pattern] = re.findall(orderPatterns[pattern], textList.lower())
    else:
        print("Unsupported format, please pass a list/Series of string, or a string.")
        out = None
        
    return out
    
def checkPattern(patterns, string, binary = True):
    '''
    Find all the occurences of each pattern in the list 'patterns' in the string.
    Parameters:
        patterns: a list/Series of patterns to be searched for
        string: the string to be searched in
    Returns:
        A list of all the matches found for each of the patterns in the list 'patterns'.
    '''
    if binary:
        return all([[]!=re.findall(pat, string.lower()) for pat in patterns])
    else:
        return [re.findall(pat, string.lower()) for pat in patterns]

def findTags(df, binary = False, filename=''):
    '''
    Finds relevant tags for the given text base don keywords.
    Parameters:
        df - list/series of texts
        filename - Name of the csv file that the search results are written to. If empty string, no file is written.
    Returns:
        A DataFrame containing the text and matches found for each tag.
    '''
#    patDate = '\d{1,2}[\\-\./]\d{1,2}(?=,|\.|\s)|\d{1,2}[\\-\./]\d{1,2}[\\-\./]\d{2,4}(?=,|\.|\s)'
    patFoc = ['foc\s+date|delivery\s+date|foc\sof']
    patCircuit = ['circuit|\slec\s|\slecs\s']
    patSiteAccess = ['access','\ssite|premises|\sroof|\sdoor|\sgate|building']
    patInstallation = ['install|schedul']
    patEscalation = ['escalat|expedit']
    patQuestion = ['where|may i|could|\swant|how|please|any update|confus|\scan\s|do you|do we|make sure|request|\?\s|need to|needs to']
    
    patAll = {'FOC Inquiry':patFoc, 'Circuit/LEC':patCircuit, 'Site Access':patSiteAccess, 'Installation':patInstallation,
              'Escalation':patEscalation, 'Actionable':patQuestion}
    
    res = []
    for email in df:
        res.append(pd.Series([checkPattern(patAll[p], email, binary=binary) for p in patAll.keys()]))
        
    patternResults = pd.DataFrame(res)
    patternResults.columns = list(patAll.keys())
    if type(df)==list:
        resultsDf = pd.concat([pd.Series(df), patternResults], axis=1, ignore_index=True)
    else:    
        resultsDf = pd.concat([df.reset_index(drop=True), patternResults], axis=1, ignore_index=True)
    resultsDf.columns = ['Email']+list(patternResults.columns)
    if filename!='':
        resultsDf.to_csv(filename+".csv", index=False)
    return resultsDf

    
    
    
    
    
    
    
    
    
    
    
    
    
    