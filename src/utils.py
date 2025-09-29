from src.logger import logger
from src.exception import CustomException
import os, sys
import pickle

def save_object(obj:object, file_path:str):
    try:
        logger.info(f"Entered save object function at {os.getcwd()}")
        if not os.path.exists(os.path.dirname(file_path)):
            logger.exception(f"The file path {file_path} does not exist")
            raise Exception(f"The file path {file_path} does not exist")
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logger.info(f"Saved {obj} at {file_path} exiting function")
    except Exception as e:
        logger.exception(CustomException(e,sys))
        raise CustomException(e,sys)

