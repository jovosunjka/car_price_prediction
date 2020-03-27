from abc import abstractmethod

from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.algorithm import BaseAlgorithm

random_state = 42



class AlgorithmWithGridSearchCV(BaseAlgorithm):

    def __init__(self, data_path_or_data, unique_values_per_columns, existing_parameters):
        self.existing_parameters = existing_parameters
        self.grid_search = None
        super().__init__(data_path_or_data, unique_values_per_columns)

    @abstractmethod
    def initialize_model_and_scaler(self):
        pass

    """ Override """
    def fit(self):
        if self.existing_parameters == None:
            self.grid_search.fit(self.x_train, self.y_train)
            print("Best score:", self.grid_search.best_score_)
            print("Best params:", self.grid_search.best_params_)
        else:
            super().fit()

    """ Override """
    def predict(self, data):
        x = self.mapCarDataToNumericValues(data)
        x = self.scaler.transform(x)
        if self.existing_parameters == None:
            y_predict = self.grid_search.predict(x)
        else:
            y_predict = self.model.predict(x)
        return y_predict

    """ Override """
    def rmse(self):
        if self.existing_parameters == None:
            super().rmse(self.grid_search)
        else:
            super().rmse(self.model)


    """ Override """
    def r2(self):
        if self.existing_parameters == None:
            super().r2(self.grid_search)
        else:
            super().r2(self.model)