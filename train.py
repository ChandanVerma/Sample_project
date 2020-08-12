import warnings
warnings.filterwarnings('ignore')
import sys
import os
src_path = os.path.abspath(os.path.join(os.getcwd() + '/src/' ))
sys.path.append(src_path)
from src.train_pipeline import *

def main():
    training_pipeline = Training_pipeline()
    training_pipeline.load_data_pipeline()
    training_pipeline.identify_data_types_pipeline()
    training_pipeline.continuous_variable_pipeline()
    training_pipeline.categorical_variable_pipeline()
    training_pipeline.discrete_variable_pipeline()
    training_pipeline.aggregate_columns_pipeline()
    training_pipeline.perform_feature_selection_pipeline()
    training_pipeline.model_training_pipeline()
    training_pipeline.model_evaluation_pipeline()
    training_pipeline.save_model_pipeline()

if __name__ == '__main__':
    main()