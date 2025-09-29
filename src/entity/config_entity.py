import sys, os

from src.constants import data_ingestion_const
class DataIngestionConfig:
    def __init__(self):
        self.artifact_dir = data_ingestion_const.ARTIFACT_DIR
        self.csv_data_access = data_ingestion_const.CSV_DATA_ACCESS
        self.train_data_path = data_ingestion_const.TRAIN_DATA_PATH
        self.test_data_path = data_ingestion_const.TEST_DATA_PATH
        self.raw_data_path = data_ingestion_const.RAW_DATA_PATH
