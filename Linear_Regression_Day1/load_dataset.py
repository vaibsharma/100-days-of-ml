# Load the dataset from Kaggle
# About the dataset - https://www.kaggle.com/datasets/amjadzhour/car-price-prediction/data
# Features - 
# Make - The make of the car
# Model - The model of the car
# Year - The year of the car
# EngineSize - The engine size of the car
# Mileage - The mileage of the car
# FuelType - The fuel type of the car
# Transmission - The transmission of the car
# Price - The price of the car

import torch
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter



CSV_FILE_NAME = "Car_Price_Prediction.csv"

class CarPriceDatasetError(Exception):
    pass

# Write a class to load the dataset
# For string columns, we will encode the data to index value
# For numerical columns, we will keep the data as is
# We will return the tensor data
class CarPriceDataset(object):
    def __init__(self, convert_to_tensor=False):
        self.input_csv_path = CSV_FILE_NAME
        self.data = None
        self.tensor_data = None
        self.convert_to_tensor = convert_to_tensor
        self.unique_columns_map = {}
        self.string_columns = []

    def is_string_column(self, column_name):
        return column_name in self.string_columns

    def load_data(self):
        if self.data is None:
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "amjadzhour/car-price-prediction",
                self.input_csv_path,
            ) 
            self.data = df

    def encoded_data(self, column_name, data):
        # Map the data to index value in the unique_columns_map
        # and return the encoded data
        encoded_data = []
        unique_values = self.unique_columns_map[column_name].tolist()  # Convert to list
        for value in data:
            # find the index of the value in the unique_columns_map
            index = unique_values.index(value)
            encoded_data.append(index+1)
        return encoded_data

    def get_data(self):
        if self.data is None:
            raise CarPriceDatasetError("Data not loaded")
        
        if self.convert_to_tensor:
            if self.tensor_data is None:
                self.string_columns = self.data.select_dtypes(include=['object']).columns

                for col in self.string_columns:
                    self.unique_columns_map[col] = self.data[col].unique()

                encoded_data = []
                for col in self.data.columns:
                    if self.is_string_column(col):
                        encoded_data.append(self.encoded_data(col, self.data[col]))
                    else:
                        encoded_data.append(self.data[col])
                
                self.tensor_data = torch.tensor(encoded_data, dtype=torch.float32)

                # Transpose the tensor data
                self.tensor_data = self.tensor_data.T
            return self.tensor_data
        
        return self.data