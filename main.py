import sys, os
from src.components.data_ingestion import DataIngestion

if __name__=='__main__':
    a = DataIngestion()
    b = a.initiate_data_ingestion()
    print(b)