import sys, os
from src.exception import CustomException
from src.logger import logger
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact
from src.utils import save_object

class DataTransformation:
    def __init__(self, data_ingestion_artifacts:DataIngestionArtifact):
        self.data_ingestion_artifacts = data_ingestion_artifacts
        self.data_transformation_config = DataTransformationConfig()
        os.makedirs(self.data_transformation_config.transformed_dir, exist_ok=True)
        logger.info(f"Created {self.data_transformation_config.transformed_dir}")
    
    def get_preprocessor_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            logger.info(f"Categorical Columns: {categorical_columns}")
            logger.info(f"Numerical Columns: {numerical_columns}")

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehotencoding", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            logger.info("Preprocessor Created!")
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self):
        try:
            logger.info("Entering Data Transformation Method")

            train_df = pd.read_csv(self.data_ingestion_artifacts.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifacts.test_file_path)
            logger.info("Train and Test Dataframes are loaded")

            preprocessor_object = self.get_preprocessor_object()
            logger.info("Got Data Transformation Object")
            target_column_name = self.data_transformation_config.target_column

            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            transformed_input_feature_train_array = preprocessor_object.fit_transform(input_feature_train_df)
            transformed_input_feature_test_array=preprocessor_object.transform(input_feature_test_df)
            logger.info("Transformed Input Train and Test Arrays")
            save_object(obj=preprocessor_object,file_path=self.data_transformation_config.preprocessor_object_file_path)

            train_array = np.c_[transformed_input_feature_train_array, np.array(target_feature_train_df)]
            test_array = np.c_[transformed_input_feature_test_array, np.array(target_feature_test_df)]

            logger.info("Exiting Data Transformation Method")
            return DataTransformationArtifact(
                train_array=train_array,
                test_array=test_array,
                preprocessor_path=self.data_transformation_config.preprocessor_object_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)


