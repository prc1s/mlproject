import sys, os

#Data Transformation Config Constants
from src.constants.general import ARTIFACT_DIR, TARGET_COLUMN
TRANSFORMATION_DIR = os.path.join(ARTIFACT_DIR, "transformation")
PREPROCESSOR_OBJECT_FILE_PATH = os.path.join(TRANSFORMATION_DIR, "preprocessor.pkl")

NUMERICAL_COLUMNS: list =  ["writing_score", "reading_score"]
CATEGORICLE_COLUMNS: list = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]