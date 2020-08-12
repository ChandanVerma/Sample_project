import yaml
import os
import sys
root = os.path.abspath('..')
sys.path.append(root)
import matplotlib.pyplot as plt
import pandas as pd

def load_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.load(f)
    return config

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def drop_dplicate_columns(df):
    df = df.T.drop_duplicates().T
    return df

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def create_missing_values_json(df, vars):
    df = df.copy()
    mv_json = {}
    for var in vars:
        mv_json[var] = df[var].isnull().sum()/len(df)
    return mv_json

def analyze_continuous(df, var):
    df = df.copy()
    df[var].hist(bins = 20)

    plt.xlabel(var)
    plt.ylabel('Number of houses')
    plt.tight_layout()
    #plt.savefig(os.path.join(config['PATH']['ANALYSIS_REPORTS_PATH'] + 'continuous_vars', '{}_distribution.png'.format(var)))
    plt.show()

def find_freq_labels(df, var, rare_pct= 0.90):
    df = df.copy()
    tmp = df[var].value_counts(normalize = True)
    return tmp[tmp > rare_pct].index
