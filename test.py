import warnings
warnings.filterwarnings('ignore')
import sys
import os
src_path = os.path.abspath(os.path.join(os.getcwd() + '/src/' ))
sys.path.append(src_path)
from src.test_pipeline import *

def main():
    test_pipeline = Test_pipeline()
    test_pipeline.load_data_pipeline()
    test_pipeline.feature_selection()
    test_pipeline.load_model_pipeline()
    test_pipeline.identify_data_types_pipeline()
    test_pipeline.continuous_variable_pipeline()
    test_pipeline.categorical_variable_pipeline()
    test_pipeline.discrete_variable_pipeline()
    test_pipeline.make_predictions()
    test_pipeline.generating_csv_file()
    test_pipeline.calculating_accuracy()

if __name__ == '__main__':
    main()