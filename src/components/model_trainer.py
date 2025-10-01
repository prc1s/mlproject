import sys, os
from src.exception import CustomException
from src.logger import logger
from src.utils import save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact
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
    def __init__(self, data_transformation_artifacts:DataIngestionArtifact):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = ModelTrainerConfig()
        os.makedirs(self.model_trainer_config.model_trainer_dir)
        logger.info(f"Created {self.model_trainer_config.model_trainer_dir}")

    def initiate_model_trainer(self):
        logger.info("Entering Model Trainer Method")
