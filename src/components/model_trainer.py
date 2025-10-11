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
from sklearn.model_selection import GridSearchCV
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

    def track_models(self,model_name, model, metrics:TestMetricsArtifacts, params):
        try:
            mlflow.set_experiment("All Models")
            with mlflow.start_run(run_name=f"{model_name}"):
                r2 = metrics.r2_score
                mean_sqrt_error = metrics.mean_squared_error
                mean_abs_error = metrics.mean_absolute_error

                mlflow.log_params(params)
                
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mean_squared_error", mean_sqrt_error)
                mlflow.log_metric("mean_absolute_error", mean_abs_error)
                
                mlflow.sklearn.log_model(model,f"{model_name}")
        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)
        

    def evaluate_model(self, x_train, y_train, x_test, y_test, models, params):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                para = params[list(models.keys())[i]]
                gs = GridSearchCV(model, para, cv=4, verbose=1)
                gs.fit(x_train, y_train)
                model_name = list(models.keys())[i]

                model.set_params(**gs.best_params_)
                model.fit(x_train, y_train)

                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                train_model_score = r2_score(y_train,y_train_pred)
                test_model_score = r2_score(y_test,y_test_pred)


                model_test_metrics = self.model_metrics(y_test, y_test_pred)
                logger.info("Got Model Metrics")

                report[list(models.keys())[i]] = test_model_score
                self.track_models(model_name, model, model_test_metrics, gs.best_params_)
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

            params = {
                "Linear Regression": {},
                
                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    "max_iter": [1000, 5000, 10000]
                },
                
                "Ridge": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]
                },
                
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "p": [1, 2]
                },
                
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                
                "Random Forest Regressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "bootstrap": [True, False]
                },
                
                "XGBRegressor": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.7, 0.8, 1.0],
                    "colsample_bytree": [0.7, 0.8, 1.0]
                },
                
                "CatBoosting Regressor": {
                    "iterations": [100, 200, 300],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 7, 9]
                },
                
                "AdaBoost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1]
                }
            }


            model_report: dict = self.evaluate_model(x_train,y_train,x_test,y_test,models,params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]
            logger.info("Best model found")

            preprocessing_obj=load_object(self.data_transformation_artifacts.preprocessor_path)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )
            y_test_pred = best_model.predict(x_test)
            r2 = r2_score(y_test,y_test_pred)
            return r2


        except Exception as e:
            logger.exception(CustomException(e,sys))
            raise CustomException(e,sys)

