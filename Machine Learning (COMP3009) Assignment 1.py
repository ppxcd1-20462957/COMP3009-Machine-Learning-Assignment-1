import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import sklearn
import keras

class Preprocess:
    def __init__(self, csv):
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.data_manager(csv)

    def data_manager(self, csv):
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

        return train_X, train_y, test_X, test_y

    def normalise(self, input_features):
        #set column values between 0-1 independently
        x = input_features

        for i in x.columns:
            x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())

        return x

#four exmaple classes depending on the models we use
class DecisionTreeReg:
    def __init__(self, train_X, train_Y, test_X, test_Y):
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import cross_val_score
    decision_regressor = DecisionTreeRegressor(random_state=44)
    decision_regressor.fit(train_X, train_Y)
    cross_val = cross_val_score(decision_regressor, train_X, train_Y, scoring='neg_mean_squared_error', cv=10)
    print("Cross validation score:", cross_val)
    pred_Y = decision_regressor.predict(test_X)
    self.calculate_mse(pred_Y, test_Y)
  
  def convert_to_numpy (self, train_X, train_Y, test_X, test_Y):
    train_X = train_X.to_numpy()
    train_Y = train_Y.to_numpy()
    test_X = test_X.to_numpy()
    test_Y = test_Y.to_numpy()

    return train_X, train_Y, test_X, test_Y
  
  def calculate_mse(self, Pred_Y, test_Y):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(Pred_Y, test_Y)
    print("MSE:", mse)

class MLP_sklearn:
  def __init__(self, train_X, train_Y, test_X, test_Y):
    
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score

    train_X, train_Y, test_X, test_Y = self.convert_to_numpy(train_X, train_Y, test_X, test_Y)
    mlp_model = MLPRegressor(hidden_layer_sizes=(8,16),activation="relu" ,random_state=42, max_iter=1000).fit(train_X, train_Y)
    cross_val = cross_val_score(mlp_model, train_X, np.ravel(train_Y), scoring='neg_mean_squared_error', cv=10)
    print("Cross validation score:", cross_val)

    pred_Y=mlp_model.predict(test_X)
    self.calculate_mse(pred_Y, test_Y)
    
  def convert_to_numpy (self, train_X, train_Y, test_X, test_Y):
    train_X = train_X.to_numpy()
    train_Y = train_Y.to_numpy()
    test_X = test_X.to_numpy()
    test_Y = test_Y.to_numpy()

    return train_X, train_Y, test_X, test_Y

  def calculate_mse(self, Pred_Y, test_Y):
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(Pred_Y, test_Y)
    print("MSE:", mse)

class Model3:
    pass

class Model4:
    pass

if __name__ == '__main__':
    wine_path = 'D:\OneDrive\Academia\MSc Machine Learning in Science\Modules\COMP3009 Machine Learning\Submissions\Assignment 1\wine.csv'
    wine_dataset = Preprocess(wine_path)
    #Model1(wine_dataset.train_X, wine_dataset.train_Y, wine_dataset.test_X, wine_dataset.test_Y)
