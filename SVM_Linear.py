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

warnings.filterwarnings(action = 'ignore', category = DataConversionWarning) 

class Preprocess:
    def __init__(self, csv):
        encoder = LabelEncoder()
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.data_manager(csv, encoder)

    def data_manager(self, csv, encoder):
        #import file, drop rows containing N/A
        input = pd.read_csv(csv, header = 0).dropna(axis = 0)
        
        #70% training data, normalise features, split class
        training_data = input.sample(frac = 0.7, random_state = 200)
        train_X = self.normalise(training_data.iloc[:, 0:11])
        train_y = training_data.iloc[:, 11:12]
        
        #30% testing data, normalise features, split class
        testing_data = input.drop(training_data.index)
        test_X = self.normalise(testing_data.iloc[:, 0:11])
        test_y = testing_data.iloc[:, 11:12]

        train_y = encoder.fit_transform(train_y)
        test_y = encoder.fit_transform(test_y)
        train_X.to_numpy()
        test_X.to_numpy()

        return train_X, train_y, test_X, test_y

    def normalise(self, input_features):
        #set column values between 0-1 independently
        x = input_features

        for i in x.columns:
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())

        return x

class SVMR:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        from sklearn.svm import SVR
        from sklearn.model_selection import cross_val_score
        svm_r = SVR(C=1.0, kernel='rbf',degree=3,gamma='scale')
        #svm_r = SVR(C=1.0, kernel='rbf',degree=3)
        #svm_r = SVR(C=1.0, kernel='rbf')
        #svm_r.fit(train_X, train_Y.ravel())
        svm_r.fit(train_X, train_Y)
        svm_r_score = svm_r.score(test_X, test_Y)
        print('The SVM accuracy score is {:03.2f}'.format(svm_r_score))
        svm_pred_Y = svm_r.predict(test_X)
        #n_folds = 5
        cross_valSVM = np.average(cross_val_score(svm_r,train_X, np.ravel(train_Y),scoring='neg_mean_squared_error', cv=10))
        print("Cross validation score:", cross_valSVM)
        pred_Y = svm_r.predict(test_X)
        #print(pred_Y)
        self.calculate_mse(pred_Y, test_Y)
    def calculate_mse(self, Pred_Y, test_Y):
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(Pred_Y, test_Y)
        print("MSE:", mse)

class LinR:
    def __init__(self, train_X, train_Y, test_X, test_Y):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score
        linear = LinearRegression().fit(train_X, train_Y)
        #linear_score = linear.score(train_X, train_Y)
        linear_score = linear.score(test_X, test_Y)
        print('The Linear regression accuracy score is {:03.2f}'.format(linear_score))
        linear_predict = linear.predict(test_X)
        #n_folds = 5
        cross_val_linear = np.average(cross_val_score(linear,train_X,train_Y,scoring='neg_mean_squared_error', cv=10))
        print("Cross validation score:", cross_val_linear)
        pred_Y = linear.predict(test_X)
        self.calculate_mse(pred_Y, test_Y)
  
    def calculate_mse(self, Pred_Y, test_Y):
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(Pred_Y, test_Y)
        print("MSE:", mse)

if __name__ == '__main__':
    #wine_path = r'C:\Users\Kshitij Tiwari\OneDrive\MLiS stuff\wine.csv'
    wine_path = 'https://raw.githubusercontent.com/ppxcd1-20462957/COMP3009-Machine-Learning-Assignment-1/master/wine.csv?token=GHSAT0AAAAAAB2LY5YRBLKJIADQ2MPPAJT6Y3QPPZA'
    print(wine_path)
    wine_dataset = Preprocess(wine_path)
    print(wine_dataset)
    SVMR(wine_dataset.train_X, wine_dataset.train_Y, wine_dataset.test_X, wine_dataset.test_Y)
    LinR(wine_dataset.train_X, wine_dataset.train_Y, wine_dataset.test_X, wine_dataset.test_Y)

