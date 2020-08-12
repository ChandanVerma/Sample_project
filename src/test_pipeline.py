import os
import sys
root = os.path.abspath('..')
sys.path.append(root)
from src.data_analysis import *
from src.feature_selection import *
from src.model import *
from src.utils import *
import logging as log
from tqdm import tqdm

class Test_pipeline:
    def __init__(self) -> None:
        self.x = 1

    def load_data_pipeline(self):
        print('<<<<<< loading data >>>>>>>')
        self.config = load_config('src/config.yaml')
        self.data = load_data(os.path.join(self.config['PATHS']['data_path'] , self.config['test_file_name']))
        self.final_data = self.data.copy() 
        print('<<<<<< data loading sucessfully completed >>>>>>>')

    def feature_selection(self):
        print('<<<<<< Loading required features >>>>>>>')
        self.final_selected_features = load_data(os.path.join(self.config['PATHS']['data_path'] , self.config['final_feature_list']))
        self.final_selected_features = list(self.final_selected_features['0'])
        self.target = self.data['y']
        del self.data['y']
        #self.final_selected_features.remove('y')
        print('<<<<<< Feature list loaded successfully >>>>>>>')

    def load_model_pipeline(self):
        print('<<<<<< Loading Model >>>>>>>')
        self.train_model = model_training(self.data, self.target, self.final_selected_features, self.config)
        self.xgb_model = self.train_model.load_model(os.path.join(self.config['PATHS']['models_path'] , self.config['model_name']))
        print('<<<<<< Model sucessfully loaded >>>>>>>')

    def identify_data_types_pipeline(self):
        print('<<<<<< data loading sucessfully completed >>>>>>>')
        self.cont_columns, self.discrete_columns, self.cat_columns = identify_data_types(self.data, self.config)

    def continuous_variable_pipeline(self):
        print('<<<<<< Processing continuous variables >>>>>>>')
        continuous_var_analysis_ = continuous_var_analysis(self.data, self.cont_columns, self.config)
        self.data = continuous_var_analysis_.impute_missing_values(method = 'mean') 
        print('<<<<<< Continuous variables processed >>>>>>>')

    def categorical_variable_pipeline(self):
        print('<<<<<< Processing Categorical variables >>>>>>>')           
        self.cat_analysis = categorical_var_analysis(self.data, self.cat_columns, self.config)
        self.data = self.cat_analysis.impute_missing_value(self.cat_columns)
        print('<<<<<< Categorical variables processed >>>>>>>')

    def discrete_variable_pipeline(self):
        print('<<<<<< Processing Categorical variables >>>>>>>')  
        self.discrete_remove = self.cat_analysis.removing_variables_with_single_value(self.discrete_columns)
        self.data = self.cat_analysis.impute_missing_value(self.discrete_columns)
        print('<<<<<< Processing Discrete variables >>>>>>>')  

    def make_predictions(self):
        print('<<<<<< Making final predictions >>>>>>>')  
        self.y_pred = self.train_model.evaluate_model()

    def generating_csv_file(self):
        print('<<<<<< Generating csv file >>>>>>>') 
        self.final_data['predictions'] = self.y_pred
        self.final_data.to_csv(os.path.join(self.config['PATHS']['data_path'] , self.config['predictions_file_name']), index = False) 

    def calculating_accuracy(self):
        print('<<<<<< Calculating accuracy >>>>>>>') 
        for i in tqdm(range(len(self.final_data))):                
            if abs(self.final_data.loc[i, 'y'] - self.final_data.loc[i, 'predictions']) <= 3:
                self.final_data.loc[i, 'accuracy'] = 1
            else:
                self.final_data.loc[i, 'accuracy'] = 0
        self.accuracy = self.final_data['accuracy'].sum() / len(self.final_data)
        print('Accuracy of the model is: ', np.round(self.accuracy, 3))




    


    


