import os
import sys
root = os.path.abspath('..')
sys.path.append(root)
import pandas as pd
from src.utils import analyze_continuous, find_freq_labels, load_config, drop_dplicate_columns, get_top_abs_correlations
from src.utils import create_missing_values_json
from src.utils import find_freq_labels
import sweetviz as sv
import missingno as msno 
import matplotlib.pyplot as plt
import json
import numpy as np

def eda(df, config):
    advert_report = sv.analyze(df, pairwise_analysis='on')
    advert_report.show_html('eda_document.html')

def identify_data_types(df, config):
    cont_columns = [col for col in df.columns if df[col].nunique() >= 18]

    discrete_columns = [col for col in df.columns if df[col].nunique() < 18 and df[col].nunique() > 5]

    cat_columns = [col for col in df.columns if df[col].nunique() <= 5]

    pd.Series(cont_columns).to_csv(os.path.join(config['PATHS']['data_path'] , 'cont_columns.csv'), columns = None, index = False)
    pd.Series(discrete_columns).to_csv(os.path.join(config['PATHS']['data_path'] , 'discrete_col.csv'), columns = None, index = False)
    pd.Series(cat_columns).to_csv(os.path.join(config['PATHS']['data_path'] , 'cat_columns.csv'), columns = None, index = False)

    return cont_columns, discrete_columns, cat_columns

def create_columns_list(cat_columns, cont_columns, discrete_columns, config):
    final_list = cat_columns + cont_columns + discrete_columns
    pd.Series(final_list).to_csv(os.path.join(config['PATHS']['data_path'] , config['feature_list_before_fs']), index = False)


class continuous_var_analysis(object):
    def __init__(self, df, cont_columns, config):
        self.df = df
        self.cont_columns = cont_columns
        self.config = config

    def plot_distribution(self):
        for var in self.cont_columns:
            analyze_continuous(self.df, var)
    
    def plot_missing_value(self):
        msno.matrix(self.df[self.cont_columns]) 
        plt.show()
    
    def get_missing_value_info(self, save_file_name = 'cont_miss_percentage'):
        mv_json = create_missing_values_json(self.df, self.cont_columns)
        with open(os.path.join(self.config['PATHS']['Project_path'] + '/data', f'{save_file_name}.json'), 'w') as f:
            json.dump(mv_json, f)

    def impute_missing_values(self, method = 'mean'):
        if method == 'mean':
            self.df[self.cont_columns] = self.df[self.cont_columns].fillna(self.df[self.cont_columns].mean())
        return self.df

    def correlation_analysis(self, values_to_display = None):
        correlated_pairs = pd.DataFrame(get_top_abs_correlations(self.df[self.cont_columns], 5000)).reset_index()
        correlated_pairs.columns = ['var_1', 'var_2', 'correlation']
        print(f'Showing top {values_to_display} values of correlated pairs')
        print(correlated_pairs.head(values_to_display))
        correlated_pairs.to_csv(os.path.join(self.config['PATHS']['Project_path'] + '/data', 'correlated_pairs.csv'), index = False)

    def removing_correlated_columns(self, correlation_threshold = 0.7):
        correlation_matrix = self.df[self.cont_columns].corr()
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))
        upper.to_csv(os.path.join(self.config['PATHS']['Project_path'] + '/data', 'corr_matrix_.csv'))
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        return to_drop

class categorical_var_analysis(object):
    def __init__(self, df, cat_columns, config) -> None:
        self.df = df
        self.cat_columns = cat_columns
        self.config = config

    def removing_variables_with_single_value(self, var_list):
        col = find_freq_labels(self.df, self.cat_columns)
        return list(col.names)

    def impute_missing_value(self, var_list, method = 'mode'):
        if method == 'mode':
            self.df[var_list]= self.df[var_list].fillna(self.df.mode().iloc[0])
        return self.df

# if __name__ == '__main__':
#     ## Load config and data   
#     config = load_config('config.yaml')
#     df = pd.read_csv(os.path.join(config['PATHS']['Project_path'] + 'data/', 'train.csv'))
    
#     ## Perform EDA
#     eda(df)

#     ## Identify different datatypes
#     cont_columns, discrete_columns, cat_columns = identify_data_types(df, config)

#     ## Analyze continuous variable/impute missing values/remove high correlated columns
#     cont_analysis = continuous_var_analysis(df, cont_columns, config)
#     df = cont_analysis.impute_missing_values()
#     cont_columns_to_drop = cont_analysis.removing_correlated_columns()
#     cont_columns = list(set(cont_columns) - set(cont_columns_to_drop))

#     ## Analyze categorical columns and impute missing values/remove unique variables/
#     cat_analysis = categorical_var_analysis(df, cat_columns, config)
#     cat_remove = cat_analysis.removing_variables_with_single_value(cat_columns)
#     cat_columns = list(set(cat_columns) - set(cat_remove))
#     df = cat_analysis.impute_missing_value(cat_columns)

#     ## Analyze discrete columns and impute missing values/remove unique variables/
#     discrete_remove = cat_analysis.removing_variables_with_single_value(discrete_columns)
#     discrete_columns = list(set(discrete_columns) - set(discrete_remove))
#     df = cat_analysis.impute_missing_value(discrete_columns)

#     ## Joining all the required columns after removal
#     create_columns_list(cat_columns, cont_columns, discrete_columns, config)




