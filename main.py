import sys, os
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    data_ingestion = DataIngestion()
    data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
    data_transformation = DataTransformation(data_ingestion_artifacts)
    data_transformation_artifacts = data_transformation.initiate_data_transformation()
    print(data_transformation_artifacts)
    model_trainer = ModelTrainer(data_transformation_artifacts)
    r2_score = model_trainer.initiate_model_trainer()
    