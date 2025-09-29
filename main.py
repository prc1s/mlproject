import sys, os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

if __name__=='__main__':
    data_ingestion = DataIngestion()
    data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation(data_ingestion_artifacts)
    data_transformation_artifacts = data_transformation.initiate_data_transformation()
    print(data_transformation_artifacts)
    