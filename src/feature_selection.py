import os
import sys
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

def feature_selection(df, var_list, target):
    X_train, y_train = df[var_list], target
    sel_ = SelectFromModel(Lasso(alpha = 0.005, random_state = 0))
    sel_.fit(X_train, y_train)
    selected_feat = X_train.columns[(sel_.get_support())]
    return selected_feat

def save_features_after_feature_selction(col_names, config):
    pd.Series(col_names).to_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'final_features.csv'))



# if __name__ == '__main__':   
#     config = load_config('config.yaml')
#     df = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'train.csv'))
#     df.head()
#     target = df['y']
#     del df['y']
#     selected_features = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'final_features_before_fs.csv'))
#     selected_features = list(selected_features['0'])
#     selected_features.remove('y')
#     final_selected_features = feature_selection(df, selected_features, target)
#     save_features_after_feature_selction(final_selected_features, config)
#     #pd.Series(final_selected_features).to_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'final_features.csv'))

