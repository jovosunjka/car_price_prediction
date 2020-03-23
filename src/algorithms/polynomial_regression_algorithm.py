import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from src import data_preprocessing

from datetime import datetime

from src.algorithms.algorithm import BaseAlgorithm

random_state = 42

class PolynomialRegressionAlgorithm(BaseAlgorithm):
    def __init__(self, data_path, unique_values_per_columns):
        super().__init__(data_path, unique_values_per_columns)

    """ Implementation of abstract method """

    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    " " " Override " " "
    def fit(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.x_train = PolynomialFeatures(degree=3, include_bias=False).fit_transform(self.x_train)
        self.x_test = PolynomialFeatures(degree=3, include_bias=False).fit_transform(self.x_test)
        self.model.fit(self.x_train, self.y_train)