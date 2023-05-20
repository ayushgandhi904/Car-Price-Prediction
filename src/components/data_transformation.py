import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder 

#Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats", "preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("Data Transformation initiated")
            num_cols = ["Levy", "Prod. year", "Engine volume", "Cylinders", "Doors", "Airbags"]
            lab_cols = ["Gear box type", "Drive wheels", "Fuel type", "Turbo engine", "Leather interior", "Wheel"]
            onehot_cols = ["Manufacturer", "Model", "Category", "Color"]
            
            gear_type = ["Manual", "Automatic", "Tiptronic", "Variator"]
            drive_wheels = ["Rear", "Front", "4x4"]
            fuel_type = ["LPG", "CNG", "Diesel" , "Petrol" , "Hybrid", "Plug-in Hybrid"]
            turbo_engine = ["No", "Yes"]
            leather_interior = ["No", "Yes"]
            wheel = ["Left wheel", "Right-hand drive"]
            
            logging.info("Pipeline started")
            #Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy= "median")),
                    ("scaler", StandardScaler())
                ]
            )

            #Label Pipeline
            lab_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("label", OrdinalEncoder(categories=[gear_type, drive_wheels, fuel_type, turbo_engine, leather_interior, wheel])),
                    ("scaler", StandardScaler())
                ]
            )

            #One Hot Encode Pipeline

            onehot_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("onehot", OneHotEncoder())
                ]
            )

            #Creating the column Transformer

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("lab_pipeline", lab_pipeline, lab_cols),
                ("onehot_pipeline", onehot_pipeline, onehot_cols)
            ])
            return preprocessor
            logging.info("Pipeline completed")
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
            
    def initiate_data_transfomation(self, train_path, test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'price'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            

            from scipy.sparse import csr_matrix

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            X_train = pd.DataFrame(X_train.toarray())
            X_test = pd.DataFrame(X_test.toarray())

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )
            logging.info('Preprocessor pickle file saved')

            return (
                X_train,
                X_test,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            logging.info("Error in Initiate Transformation")
            raise CustomException(e, sys)
            
    
