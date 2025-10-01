import sys, os
from src.constants.general import ARTIFACT_DIR
MODEL_TRAINER_DIR = os.path.join(ARTIFACT_DIR, "model_trainer")
TRAINED_MODEL_FILE_PATH = os.path.join(MODEL_TRAINER_DIR, "model.pkl")