import os
import sys
root = os.path.abspath('..')
sys.path.append(root)
from src.data_analysis import *
from src.feature_selection import *
from src.model import *
from src.utils import *
import logging as log

class Training_pipeline:
    def __init__(self) -> None:
        self.x = 1
        
    def load_data_pipeline(self):
        print('<<<<<< loading data >>>>>>>')
        self.config = load_config('src/config.yaml')
        self.data = load_data(os.path.join(self.config['PATHS']['data_path'], self.config['train_file_name'])) 
        print('<<<<<< data loading sucessfully completed >>>>>>>')

    def identify_data_types_pipeline(self):
        print('<<<<<< Identifying datatypes >>>>>>>')
        self.cont_columns, self.discrete_columns, self.cat_columns = identify_data_types(self.data, self.config)

    def continuous_variable_pipeline(self):
        print('<<<<<< Processing continuous variables >>>>>>>')
        continuous_var_analysis_ = continuous_var_analysis(self.data, self.cont_columns, self.config)
        self.data = continuous_var_analysis_.impute_missing_values(method = 'mean')   

        self.cont_columns_to_drop = continuous_var_analysis_.removing_correlated_columns()
        self.cont_columns = list(set(self.cont_columns) - set(self.cont_columns_to_drop))
        print('<<<<<< Continuous variables processed >>>>>>>')

    def categorical_variable_pipeline(self):
        print('<<<<<< Processing Categorical variables >>>>>>>')       
        self.cat_analysis = categorical_var_analysis(self.data, self.cat_columns, self.config)
        self.cat_remove = self.cat_analysis.removing_variables_with_single_value(self.cat_columns)
        self.cat_columns = list(set(self.cat_columns) - set(self.cat_remove))
        self.data = self.cat_analysis.impute_missing_value(self.cat_columns)
        print('<<<<<< Categorical variables processed >>>>>>>')

    def discrete_variable_pipeline(self):
        print('<<<<<< Processing Discrete variables >>>>>>>')  
        self.discrete_remove = self.cat_analysis.removing_variables_with_single_value(self.discrete_columns)
        self.discrete_columns = list(set(self.discrete_columns) - set(self.discrete_remove))
        self.data = self.cat_analysis.impute_missing_value(self.discrete_columns)
        print('<<<<<< Processing Discrete variables >>>>>>>')  
    
    def aggregate_columns_pipeline(self):
        print('<<<<<< Combining all required columns >>>>>>>') 
        create_columns_list(self.cat_columns, self.cont_columns, self.discrete_columns, self.config)
        print('<<<<<< Completed >>>>>>>') 

    def perform_feature_selection_pipeline(self):
        print('<<<<<< Performing feature selection >>>>>>>') 
        self.selected_features = load_data(os.path.join(self.config['PATHS']['data_path'] , self.config['feature_list_before_fs']))
        selected_features_list = list(self.selected_features['0'])
        self.target = self.data['y']
        del self.data['y']
        selected_features_list.remove('y')
        self.selected_features = selected_features_list
        self.final_selected_features = feature_selection(self.data, self.selected_features, self.target)
        save_features_after_feature_selction(self.final_selected_features, self.config)
        print('<<<<<< Feature selection completed successfully >>>>>>>') 

    def model_training_pipeline(self): 
        print('<<<<<< Training model >>>>>>>')       
        self.train_model = model_training(self.data, self.target, self.final_selected_features, self.config)
        #train_model.selecting_best_model()
        self.train_model.train_baseline_model(model_name = 'xgboost')
        print('<<<<<< Training completed successfully >>>>>>>')        

    def model_evaluation_pipeline(self):
        print('<<<<<< Evaluating model performance >>>>>>>') 
        self.train_model.evaluate_model()


    def save_model_pipeline(self):
        print('<<<<<< Saving model to models directory >>>>>>>') 
        self.train_model.save_model()
        print('<<<<<< Successfully saved model >>>>>>>') 

    def load_model_pipeline(self):
        print('<<<<<< Loading model from model directory >>>>>>>') 
        self.train_model.load_model(os.path.join(self.config['PATHS']['models_path'] , self.config['model_name']))
        print('<<<<<< Model loaded successfully >>>>>>>') 


    









