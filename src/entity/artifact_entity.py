import sys, os
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_file_path : str
    test_file_path : str

@dataclass
class DataTransformationArtifact:
    test_array: str
    train_array: str
    preprocessor_path: str

@dataclass
class TestMetricsArtifacts:
    r2_score: float
    mean_absolute_error: float
    mean_squared_error: float
@dataclass
class ModelTrainerArtifact:
    s:float