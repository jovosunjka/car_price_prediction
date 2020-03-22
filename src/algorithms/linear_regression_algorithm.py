
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.algorithms.algorithm import BaseAlgorithm


class LinearRegressionAlgorithm(BaseAlgorithm):
    def __init__(self, data_path, unique_values_per_columns):
        super().__init__(data_path, unique_values_per_columns)

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()