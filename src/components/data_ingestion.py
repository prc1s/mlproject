import os,sys
from src.exception import CustomException
from src.logger import logger 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        os.makedirs(self.data_ingestion_config.ingested_data_dir, exist_ok=True)
        logger.info(f"Created {self.data_ingestion_config.ingested_data_dir}")
    
    def initiate_data_ingestion(self):
        logger.info("Entered Data Ingestion Method")
        try:
            df = pd.read_csv(self.data_ingestion_config.csv_data_access)
            logger.info("Read Dataset as Dataframe")
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)
            logger.info("Train, Test Split Completed"+
                        f"\nTrain set at {self.data_ingestion_config.train_data_path}\nTest set at {self.data_ingestion_config.test_data_path}")
            logger.info("Exiting Data Ingestion Method")
            return DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_data_path,
                test_file_path=self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)
        

