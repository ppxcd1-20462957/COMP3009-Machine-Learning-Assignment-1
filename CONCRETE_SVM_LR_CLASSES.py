import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import DataConversionWarning 
from sklearn.metrics import accuracy_score 

#add rajat's preprocess function

class SVMR:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        from sklearn.svm import SVR
        from sklearn.model_selection import cross_val_score
        svm_r = SVR(C=1.0, kernel='poly',degree=7,gamma='scale')
        svm_r.fit(train_X, train_Y)
        svm_r_score = svm_r.score(test_X, test_Y) #R2 score
        print('The SVM accuracy score is {:03.2f}'.format(svm_r_score))
        pred_Y = svm_r.predict(test_X)
        cross_valSVM = np.average(cross_val_score(svm_r,train_X, np.ravel(train_Y),scoring='neg_mean_squared_error', cv=10))
        print("Cross validation score:", cross_valSVM)
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(test_Y, pred_Y)
        print("MSE:", mse)
    

class LinR:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        linear = LinearRegression().fit(train_X, train_Y)
        linear_score = linear.score(test_X, test_Y) #R2 score
        print('The Linear regression accuracy score is {:03.2f}'.format(linear_score))
        cross_val_linear = np.average(cross_val_score(linear,train_X,train_Y,scoring='neg_mean_squared_error', cv=10))
        print("Cross validation score:", cross_val_linear)
        pred_Y = linear.predict(test_X)
        mse = mean_squared_error(test_Y, pred_Y)
        print("MSE:", mse)

#main function 
