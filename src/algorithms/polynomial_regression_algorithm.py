
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from src.algorithms.algorithm import BaseAlgorithm

random_state = 42

class PolynomialRegressionAlgorithm(BaseAlgorithm):
    def __init__(self, data_path, unique_values_per_columns):
        self.polynomial_features = None
        super().__init__(data_path, unique_values_per_columns)

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()
        self.polynomial_features =  PolynomialFeatures(degree=4, include_bias=True)
        #self.model = LinearRegression()
        #self.model = Ridge()
        self.model = Lasso(max_iter=1000, tol=0.0001, alpha=0.001)
        #self.model = ElasticNet()

    """ Override """
    def fit(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        self.x_train = self.polynomial_features.fit_transform(self.x_train)
        self.x_test = self.polynomial_features.transform(self.x_test)

        self.model.fit(self.x_train, self.y_train)