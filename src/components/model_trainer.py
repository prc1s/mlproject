import sys, os
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object, load_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact,TestMetricsArtifacts
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,AdaBoostClassifier
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings

class ModelTrainer:
    def __init__(self, data_transformation_artifacts:DataTransformationArtifact):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)
        logger.info(f"Created {self.model_trainer_config.model_trainer_dir}")

    def model_metrics(self,y_true, y_pred):
        try:
            r2 = r2_score(y_true=y_true,y_pred=y_pred)
            mean_abs_error = mean_absolute_error(y_true=y_true,y_pred=y_pred)
            mean_sqr_error = mean_squared_error(y_pred=y_pred,y_true=y_true)
            metrics = TestMetricsArtifacts(
                r2_score=r2,
                mean_absolute_error=mean_abs_error,
                mean_squared_error=mean_sqr_error
            )
            return metrics
        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)

    def track_models(self,model_name, model, metrics:TestMetricsArtifacts):
        try:
            mlflow.set_experiment("All Models")
            with mlflow.start_run(run_name=f"{model_name}"):
                r2 = metrics.r2_score
                mean_sqrt_error = metrics.mean_squared_error
                mean_abs_error = metrics.mean_absolute_error
                
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mean_squared_error", mean_sqrt_error)
                mlflow.log_metric("mean_absolute_error", mean_abs_error)

                mlflow.sklearn.log_model(model,f"{model_name}")
        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)
        

    def evaluate_model(self, x_train, y_train, x_test, y_test, models):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model_name = list(models.keys())[i]

                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = r2_score(y_train,y_train_pred)
                test_model_score = r2_score(y_test,y_test_pred)


                model_test_metrics = self.model_metrics(y_test, y_test_pred)
                logger.info("Got Model Metrics")

                report[list(models.keys())[i]] = test_model_score
                self.track_models(model_name, model, model_test_metrics)
            return report
        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)



    def initiate_model_trainer(self):
        try:
            logger.info("Entering Model Trainer Method")
            x_train,y_train,x_test,y_test = (
                self.data_transformation_artifacts.train_array[:,:-1],
                self.data_transformation_artifacts.train_array[:,-1],
                self.data_transformation_artifacts.test_array[:,:-1],
                self.data_transformation_artifacts.test_array[:,-1]
            )

            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report: dict = self.evaluate_model(x_train,y_train,x_test,y_test,models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values).index(best_model_score)]

            best_model = models[best_model_name]
            logger.info("Best model found")

            preprocessing_obj=load_object(self.data_transformation_artifacts.preprocessor_path)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )
            y_test_pred = best_model.predict(y_test)
            r2 = r2_score(y_test,y_test_pred)
            return r2


        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)

