import os
import sys
root = os.path.abspath('..')
sys.path.append(root)
import pandas as pd
from src.utils import load_config
from sklearn.model_selection import train_test_split
import logging as log

def create_holdout_set(df, test_split_percent):
    X_train, X_test = train_test_split(df, test_size = test_split_percent, random_state = 2020)
    X_test.to_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'hold_out.csv'), index = False)
    X_train.to_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'train.csv'), index = False)

if __name__ == '__main__':
    config = load_config('config.yaml')
    df = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'dataset_00_with_header.csv'))
    create_holdout_set(df, 0.9)
    log.info('Holdout set created !!!')