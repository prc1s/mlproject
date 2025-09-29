import sys, os

#Data Ingestion Config Constants
from src.constants.general import ARTIFACT_DIR
INGESTED_DATA_DIR = os.path.join(ARTIFACT_DIR, "ingested_data")
CSV_DATA_ACCESS = os.path.join("notebook", "data", "stud.csv")
TRAIN_DATA_PATH = os.path.join(INGESTED_DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(INGESTED_DATA_DIR, "test.csv")
RAW_DATA_PATH = os.path.join(INGESTED_DATA_DIR, "data.csv")