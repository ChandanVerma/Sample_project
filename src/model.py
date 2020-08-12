import os
import sys

from xgboost.callback import early_stop
root = os.path.abspath('..')
sys.path.append(root)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import yaml
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', None)
from src.utils import load_config
from pycaret.regression import *
import xgboost as xgb
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from scipy.stats import uniform, randint
import pickle

class model_training(object):
    def __init__(self, df, target, final_columns, config) -> None:
        self.df = df
        self.target = target
        self.final_columns = final_columns
        self.config = config

    def selecting_best_model(self):
        clf1 = setup(data = df[final_columns],
                 target = target,
                 numeric_imputation= 'mean',
                 categorical_imputation = 'mode',
                 silent = True, transformation= True, transformation_method= 'yeo-johnson')

        compare_models()   

    def train_baseline_model(self, model_name = 'xgboost'):
        if model_name == 'xgboost':
            self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
            self.xgb_model.fit(self.df[self.final_columns], self.target)

    def evaluate_model(self):
        self.y_pred = self.xgb_model.predict(self.df[self.final_columns])
        mse = mean_squared_error(self.target, self.y_pred)
        print('Root mean squared error', np.sqrt(mse))
        return self.y_pred

    def display_scores(self, scores):
        print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))

    def report_best_scores(self, results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def hyperparameter_search(self, params):
        self.xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42, 
                                              early_stopping_rounds = 5)
        search = RandomizedSearchCV(self.xgb_model, 
                                    param_distributions=params, 
                                    random_state=42, n_iter=200, 
                                    cv=3, verbose=1, n_jobs=1, 
                                    return_train_score=True)
        
        search.fit(self.df[self.final_columns], self.target)

        self.report_best_scores(search.cv_results_, 1)


    def save_model(self):
        pickle.dump(self.xgb_model, open(os.path.join(self.config['PATHS']['Project_path'] + 'models/', self.config['model_name']), "wb"))
    
    def load_model(self, path):
        self.xgb_model = pickle.load(open(path, "rb"))


# if __name__ == '__main__':
#     config = load_config('config.yaml')
#     df = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'train.csv'))
#     final_variables = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'final_features.csv'))
#     target = df['y']
#     list(final_variables['0']).remove('y')
#     train_model = model_training(df, target, final_variables, config)
#     #train_model.selecting_best_model()
#     train_model.train_baseline_model(model_name = 'xgboost')
#     train_model.evaluate_model()
#     params = {
#     "colsample_bytree": uniform(0.7, 0.3),
#     "gamma": uniform(0, 0.5),
#     "learning_rate": uniform(0.03, 0.3), # default 0.1 
#     "max_depth": randint(2, 6), # default 3
#     "n_estimators": randint(100, 150), # default 100
#     "subsample": uniform(0.6, 0.4)
#     }
