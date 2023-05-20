import os, sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

#Initializing the Data Ingestion Configuration

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    


#Data ingestion class
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
        
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df = pd.read_csv(os.path.join("notebooks/data/car_price.csv"))
            df = df.drop(labels = ["ID", "Mileage"], axis = 1)
            df = df[df['Model'].isin(df['Model'].value_counts()[df['Model'].value_counts() > 100].index)]
            df["Levy"] = df["Levy"].replace("-", 0).astype(int)
            df["Turbo engine"] = df["Engine volume"].str.contains("Turbo")
            df["Turbo engine"] = df["Turbo engine"].replace({True:"Yes", False:"No"})
            df["Engine volume"] = df["Engine volume"].str.replace("Turbo", "").astype(float)
            df["Doors"]= df["Doors"].str.replace("-May", "").str.replace("-Mar", "").replace(">5", 5).astype(float)
            df = df.drop_duplicates()
            logging.info("Data set readed as Pandas DataFrame")         
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            
            logging.info("Train test split")
            train_set, test_set = train_test_split(df, test_size = 0.3, random_state = 50)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data Ingestion complted")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
            
        
        except Exception as e:
            logging.info("Exception occur at Data Ingestion step")
            raise CustomException(e, sys)
        
#Running the Data Ingestion

if __name__ == "main":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    