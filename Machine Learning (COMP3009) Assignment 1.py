import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from keras import layers, models
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings(action = 'ignore', category = DataConversionWarning)
warnings.filterwarnings(action = 'ignore', category = ConvergenceWarning)

class Preprocessing_Concrete:
    def __init__(self):
            csv = 'D:\OneDrive\Academia\MSc Machine Learning in Science\Modules\COMP3009 Machine Learning\Submissions\Assignment 1\concrete.csv'
            self.train_X, self.train_y, self.test_X, self.test_y = self.data_manager(csv)

    def data_manager(self, csv):
        input = pd.read_csv(csv)
        input.columns = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age', 'Concrete Compressive Strength']
        
        self.correlation_matrix(input)
        self.boxplots(input)

        X = input.drop('Concrete Compressive Strength', axis = 1)
        y = input['Concrete Compressive Strength']

        X = self.normalise(X)

        train_X , test_X , train_y ,test_y = train_test_split(X, y, test_size = 0.2)

        return train_X, train_y, test_X, test_y

    def correlation_matrix(self, input):
        crl_matrix = input.corr()
        mask = np.zeros_like(crl_matrix)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize = (20, 12))
        plt.title('Concrete Feature Correlation')
        cmap = sb.diverging_palette(260, 10, as_cmap = True)

        sb.heatmap(crl_matrix, vmax = 1.2, square = False, cmap = cmap, mask = mask, ax = ax, annot = True, fmt = '.2g', linewidths = 1)
        fig.savefig('heatmap_concrete.png')

    def boxplots(self, input):
        means = input.iloc[:, :]
        means.plot(kind = 'box', subplots = True, layout = (8, 4), sharex = False, sharey = False, fontsize = 12, figsize = (30, 20))

        fig, ax = plt.subplots(1, figsize = (20, 8))
        sb.boxplot(data = input.iloc[:, :],ax = ax)
        fig.savefig('boxplot_concrete.png')

    def normalise(self, X):
        min_max_scaler = MinMaxScaler()
        cols = list(X.columns.values)
        X[cols] = min_max_scaler.fit_transform(X[cols])

        return X

class Preprocessing_Wine:
    def __init__(self):
        csv = 'D:\OneDrive\Academia\MSc Machine Learning in Science\Modules\COMP3009 Machine Learning\Submissions\Assignment 1\wine.csv'
        encoder = LabelEncoder()
        self.train_X, self.train_y, self.test_X, self.test_y = self.data_manager(csv, encoder)

    def data_manager(self, csv, encoder):
        #import file, drop rows containing N/A
        input = pd.read_csv(csv, header = 0).dropna(axis = 0)
        input.head()

        sb.countplot(x = 'quality', data = input)
        self.correlation_matrix(input)
        self.boxplots(input)
        
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

    def correlation_matrix(self, input):
        crl_matrix = input.corr()
        mask = np.zeros_like(crl_matrix)
        mask[np.triu_indices_from(mask)] = True

        fig, ax = plt.subplots(figsize=(20, 12))
        plt.title('Wine quality Feature Correlation')
        cmap = sb.diverging_palette(260, 10, as_cmap=True)
        sb.heatmap(crl_matrix, vmax = 1.2, square = False, cmap = cmap, mask = mask, ax = ax, annot = True, fmt = '.2g', linewidths = 1)
        fig.savefig('heatmap_wine.png')

    def boxplots(self, input):
        means = input.iloc[:, :]
        means.plot(kind = 'box', subplots = True, layout = (8, 4), sharex = False, sharey = False, fontsize = 12, figsize = (30, 20))

        fig, ax = plt.subplots(1, figsize = (20, 8))
        sb.boxplot(data = input.iloc[:, :],ax = ax)
        fig.savefig('boxplot_wine.png')

        fig,ax=plt.subplots(1, figsize = (20, 8))
        sb.boxplot(data = input.iloc[:, :],ax=ax)

class Classification_MultilayerPerceptron:
    def __init__(self, train_X, train_y, test_X, test_y):
        model_accuracy = self.train(train_X, train_y, test_X, test_y)
        print('\nClassification: Multilayer Perceptron Model Accuracy (Per Fold): {}'.format(model_accuracy))
   
    def train(self, train_X, train_y, test_X, test_y):
        inputs = np.concatenate((train_X, test_X), axis = 0)
        targets = np.concatenate((train_y, test_y), axis = 0)
        
        kf = KFold(5, shuffle = True, random_state = 42) 
        fold = 0
        accuracy_per_fold = []

        for(train, test) in kf.split(inputs, targets):
            fold += 1
            print('Fold: {}'.format(fold))

            model = models.Sequential([
                                       layers.Dense(10, input_dim = 11, activation = 'relu'),
                                       layers.Dropout(0.1),
                                       layers.Dense(30, activation = 'relu'),
                                       layers.Dropout(0.1),
                                       layers.Dense(6, activation = 'softmax')
            ])

            model.compile(
                          loss = 'sparse_categorical_crossentropy', 
                          optimizer = 'sgd', 
                          metrics = ['accuracy']
            )

            model.fit(inputs[train], targets[train], epochs = 500)
            scores = model.evaluate(inputs[test], targets[test], verbose = 0)
            accuracy_per_fold.append('{}%'.format(np.round(scores[1] * 100, 2)))

        return accuracy_per_fold

    def cf_matrix(self, test_y, pred_y, clf):
        cm = confusion_matrix(test_y, pred_y, labels = clf.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
        display.plot()
        plt.show()

class Classification_MultinomialLogisticRegression:
    def __init__(self, train_X, train_y, test_X, test_y):
        model = self.model_init()
        model = self.train(model, train_X, train_y)
        model_accuracy = self.test(model, test_X, test_y)
        print('\nClassification: Multinomial Logistic Regression Model Accuracy: {}%'.format(100 * model_accuracy))

    def model_init(self):
        model = LogisticRegressionCV(
                                     max_iter = 2000,
                                     cv = 5,
                                     multi_class = 'multinomial', 
                                     solver = 'lbfgs', 
                                     penalty = 'l2', 
                                     )

        return model

    def train(self, model, train_X, train_y):
        model.fit(train_X, np.ravel(train_y))

        return(model)

    def test(self, model, test_X, test_y):
        pred_y = model.predict(test_X)
        self.cf_matrix(test_y, pred_y, model)
        
        return accuracy_score(test_y, pred_y)

    def cf_matrix(self, test_y, pred_y, clf):
        cm = confusion_matrix(test_y, pred_y, labels = clf.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
        display.plot()
        plt.show()

class Classification_SVM:
    def __init__(self, train_X, train_y, test_X, test_y):
        k_folds = 5
        score, cross_val = self.train(train_X, train_y, test_X, test_y, k_folds)

        print('\nClassification: Support Vector Machine Model Accuracy: {}%'.format(np.round(score * 100, 2)))
        print('Classification: Support Vector Machine {}-Fold Cross-Validation Score: {}%'.format(k_folds, np.round(cross_val * 100, 2)))
    
    def train(self, train_X, train_y, test_X, test_y, folds):
        clf = SVC(C = 1.0, kernel = 'rbf', degree = 3, gamma = 'auto', probability = True)
        clf.fit(train_X, np.ravel(train_y))
        score = clf.score(test_X, test_y)
        pred_y = clf.predict(test_X)
        cross_val = np.average(cross_val_score(clf, train_X, np.ravel(train_y), cv = folds))
        self.cf_matrix(test_y, pred_y, clf)

        return score, cross_val

    def cf_matrix(self, test_y, pred_y, clf):
        cm = confusion_matrix(test_y, pred_y, labels = clf.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
        display.plot()
        plt.show()
        
class Classification_DecisionTree:
    def __init__(self, train_X, train_y, test_X, test_y):
        k_folds = 5
        score, cross_val = self.train(train_X, train_y, test_X, test_y, k_folds)

        print('\nClassification: Decision Tree Model Accuracy {}%'.format(np.round(score * 100, 2)))
        print('Classification: Decision Tree Model {}-Fold Cross-Validation Score: {}%'.format(k_folds, np.round(cross_val * 100, 2)))

    def train(self, train_X, train_y, test_X, test_y, folds):
        clf = DecisionTreeClassifier()
        clf.fit(train_X, np.ravel(train_y))
        score = clf.score(test_X, test_y)
        pred_y = clf.predict(test_X)
        cross_val = np.average(cross_val_score(clf, train_X, np.ravel(train_y), cv = folds))
        self.cf_matrix(test_y, pred_y, clf)

        return score, cross_val

    def cf_matrix(self, test_y, pred_y, clf):
        cm = confusion_matrix(test_y, pred_y, labels = clf.classes_)
        display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = clf.classes_)
        display.plot()
        plt.show()

class Regression_MultilayerPerceptron:
    def __init__(self, train_X, train_y, test_X, test_y):
        score, cross_val, mse = self.train(train_X, train_y, test_X, test_y)
        print('\nRegression: Multilayer Perceptron Model R2 Score: {}'.format(np.round(score, 2)))
        print('Regression: Multilayer Perceptron Cross Validation MSE: {}'.format(np.round(cross_val, 2)))
        print('Regression: Multilayer Perceptron Test MSE: {}'.format(np.round(mse, 2)))

    def train(self, train_X, train_y, test_X, test_y):
        mlp = MLPRegressor(hidden_layer_sizes = (8, 16), activation = "relu", random_state = 42, max_iter = 1000)
        mlp.fit(train_X, train_y)
        score = mlp.score(test_X, test_y)
        cross_val = np.average(cross_val_score(mlp, train_X, np.ravel(train_y), scoring = 'neg_mean_squared_error', cv = 5))
        pred_y = mlp.predict(test_X)
        mse = mean_squared_error(test_y, pred_y)

        return score, cross_val, mse

class Regression_LinearRegression:
    def __init__(self, train_X, train_y, test_X, test_y):
        score, cross_val, mse = self.train(train_X, train_y, test_X, test_y)
        print('\nRegression: Linear Regression Model R2 Score: {}'.format(np.round(score, 2)))
        print('Regression: Linear Regression Cross-Validation MSE: {}'.format(np.round(cross_val, 2)))
        print('Regression: Linear Regression Test MSE: {}'.format(np.round(mse, 2)))

    def train(self, train_X, train_y, test_X, test_y):
        linear = LinearRegression()
        linear.fit(train_X, train_y)
        score = linear.score(test_X, test_y)
        cross_val = np.average(cross_val_score(linear, train_X, train_y, scoring = 'neg_mean_squared_error', cv = 5))
        pred_y = linear.predict(test_X)
        mse = mean_squared_error(test_y, pred_y)

        return score, cross_val, mse

class Regression_SVM:
    def __init__(self, train_X, train_y, test_X, test_y):
        score, cross_val, mse = self.train(train_X, train_y, test_X, test_y)
        print('\nRegression: Support Vector Machine Model R2 Score: {}'.format(np.round(score, 2)))
        print('Regression: Support Vector Machine Cross Validation MSE: {}'.format(np.round(cross_val, 2)))
        print('Regression: Support Vector Machine Test MSE: {}'.format(np.round(mse, 2)))

    def train(self, train_X, train_y, test_X, test_y):
        svr = SVR(C = 1.0, kernel = 'poly', degree = 6, gamma = 'scale')
        svr.fit(train_X, train_y)
        score = svr.score(test_X, test_y)
        pred_y = svr.predict(test_X)
        cross_val = np.average(cross_val_score(svr, train_X, np.ravel(train_y), scoring = 'neg_mean_squared_error', cv = 5))
        pred_y = svr.predict(test_X)
        mse = mean_squared_error(test_y, pred_y)

        return score, cross_val, mse

class Regression_DecisionTree:
    def __init__(self, train_X, train_y, test_X, test_y):
        score, cross_val, mse = self.train(train_X, train_y, test_X, test_y)
        print('\nRegression: Decision Tree Model R2 Score: {}'.format(np.round(score, 2)))
        print('Regression: Decision Tree Cross Validation MSE: {}'.format(np.round(cross_val, 2)))
        print('Regression: Decision Tree Test MSE: {}\n'.format(np.round(mse, 2)))

    def train(self, train_X, train_y, test_X, test_y):
        dtr = DecisionTreeRegressor(random_state = 44)
        dtr.fit(train_X, train_y)
        score = dtr.score(test_X, test_y)
        cross_val = np.average(cross_val_score(dtr, train_X, train_y, scoring='neg_mean_squared_error', cv = 5))
        pred_y = dtr.predict(test_X)
        mse = mean_squared_error(test_y, pred_y)

        return score, cross_val, mse

if __name__ == '__main__':
    wine_data = Preprocessing_Wine()
    concrete_data = Preprocessing_Concrete()

    # Classification ML
    Classification_MultilayerPerceptron(wine_data.train_X, wine_data.train_y, wine_data.test_X, wine_data.test_y)
    Classification_MultinomialLogisticRegression(wine_data.train_X, wine_data.train_y, wine_data.test_X, wine_data.test_y)
    Classification_SVM(wine_data.train_X, wine_data.train_y, wine_data.test_X, wine_data.test_y)
    Classification_DecisionTree(wine_data.train_X, wine_data.train_y, wine_data.test_X, wine_data.test_y)

    # Regression ML
    Regression_MultilayerPerceptron(concrete_data.train_X, concrete_data.train_y, concrete_data.test_X, concrete_data.test_y)
    Regression_LinearRegression(concrete_data.train_X, concrete_data.train_y, concrete_data.test_X, concrete_data.test_y)
    Regression_SVM(concrete_data.train_X, concrete_data.train_y, concrete_data.test_X, concrete_data.test_y)
    Regression_DecisionTree(concrete_data.train_X, concrete_data.train_y, concrete_data.test_X, concrete_data.test_y)