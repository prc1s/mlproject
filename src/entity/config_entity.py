import sys, os

from src.constants import data_ingestion_const, data_transformation_const
class DataIngestionConfig:
    def __init__(self):
        self.ingested_data_dir = data_ingestion_const.INGESTED_DATA_DIR
        self.csv_data_access = data_ingestion_const.CSV_DATA_ACCESS
        self.train_data_path = data_ingestion_const.TRAIN_DATA_PATH
        self.test_data_path = data_ingestion_const.TEST_DATA_PATH
        self.raw_data_path = data_ingestion_const.RAW_DATA_PATH


class DataTransformationConfig:
    def __init__(self):
        self.transformed_dir = data_transformation_const.TRANSFORMATION_DIR
        self.target_column = data_transformation_const.TARGET_COLUMN
        self.preprocessor_object_file_path = data_transformation_const.PREPROCESSOR_OBJECT_FILE_PATH