import sys, os

#Data Transformation Config Constants
from src.constants.general import ARTIFACT_DIR, TARGET_COLUMN
TRANSFORMATION_DIR = os.path.join(ARTIFACT_DIR, "transformation")
PREPROCESSOR_OBJECT_FILE_PATH = os.path.join(TRANSFORMATION_DIR, "preprocessor.pkl")

