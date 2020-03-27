
from abc import ABC, abstractmethod

from sklearn import metrics
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from src import data_preprocessing


random_state = 42

columns = ("new","price","brandName","modelName","fuelType","color","mileage","power","year","average_earning")


""" Abstract class """
class BaseAlgorithm(ABC):

    def __init__(self, data_path_or_data, unique_values_per_columns):
        self.unique_values_per_columns = unique_values_per_columns

        self.cities_and_average_earnings = None
        self.cities = None

        #self.columns_of_categorical_data = (1, 2, 4, 5, 6, 7)
        self.columns_of_categorical_data = (2, 4, 5, 6, 7)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.scaler = None
        self.model = None

        self.prepare_data(data_path_or_data)
        self.initialize_model_and_scaler()

    """
        data - dictionary or string (file path)
    """
    def prepare_data(self, data_path_or_data):
        if type(data_path_or_data) is dict:
            self.x_train = data_path_or_data["x_train"]
            self.x_train = data_path_or_data["x_test"]
            self.y_train = data_path_or_data["y_train"]
            self.y_test = data_path_or_data["y_test"]
        elif type(data_path_or_data) is str:
            data = self.load_data(data_path_or_data)
            data = self.remove_rows_with_big_values(data)

            #for i in range(len(columns)):
            #    self.plot_values_by_column(data, i)

            y = data[:, 1]  # price column
            x = np.delete(data, 1, axis=1)  # without price column

            #self.plot_values_by_column(x, y)

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                    random_state=random_state)
        else:
            raise Exception("Invalid type of 'data_path_or_data'")


    @abstractmethod
    def initialize_model_and_scaler(self):
        pass

    def remove_rows_with_big_values(self, data):
        #inexes_for_removing = self.get_indexes_for_removing_600rows(data)
        inexes_for_removing = self.get_indexes_for_removing_150k_rows(data)
        data = np.delete(data, inexes_for_removing, axis=0)
        return data

    def get_indexes_for_removing_150k_rows(self, data):
        inexes_for_removing = set()

        for index, row in enumerate(data):
            #if row[0] == 1:
            #    inexes_for_removing.add(index)
            #if row[1] >= 999999999:
            if row[1] > 80000:
            #if row[1] >= 20000:
            #if row[1] >= 10000:
            #if row[1] >= 8000:
                inexes_for_removing.add(index)

            if row[4] == 4.0 or row[4] == 5.0:
                inexes_for_removing.add(index)

            if row[6] > 500000:
                inexes_for_removing.add(index)

            if row[7] < 30 or row[7] > 625:
                inexes_for_removing.add(index)

            if row[8] <= 1995 or row[8] > datetime.now().year:
                inexes_for_removing.add(index)
            """
            if row[9] >= 115000:
                inexes_for_removing.add(index)
            """

        inexes_for_removing = list(inexes_for_removing)
        return inexes_for_removing

    def plot_values_by_column(self, data, column_index):
        print("column_index = ", column_index)
        # https://seaborn.pydata.org/generated/seaborn.boxplot.html
        #sns.boxplot(x="new", y="price", data={ "new": data[:, 1], "price": data[:, column_index]})
        sns.boxplot(data=data[:, column_index])
        plt.show()

    def fit(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model.fit(self.x_train, self.y_train)

    def get_cities_and_average_earnings(self):
        file_path_to_average_earnings_per_cities = "..\\data\\prosecne-zarade-po-opstinama.csv"
        self.cities_and_average_earnings = data_preprocessing.get_cities_and_average_earnings(file_path_to_average_earnings_per_cities)
        self.cities = list(map(lambda x: x[0], self.cities_and_average_earnings))

    def mapCarDataToNumericValues(self, data):
        x = []

        try:
            new_car = self.unique_values_per_columns[2].index(data["new_car"])
            x.append(new_car)
        except:
            raise Exception("For column 'new', not existing value:  " + data["new_car"])

        try:
            brand_name = self.unique_values_per_columns[4].index(data["brand_name"])
            x.append(brand_name)
        except:
            raise Exception("For column 'brandName', not existing value:  " + data["brand_name"])

        try:
            model_name = self.unique_values_per_columns[5].index(data["model_name"])
            x.append(model_name)
        except:
            raise Exception("For column 'modelName', not existing value:  " + data["model_name"])

        try:
            fuel_type = self.unique_values_per_columns[6].index(data["fuel_type"])
            x.append(fuel_type)
        except:
            raise Exception("For column 'fuelType', not existing value:  " + data["fuel_type"])

        try:
            color = self.unique_values_per_columns[7].index(data["color"])
            x.append(color)
        except:
            raise Exception("For column 'color', not existing value:  " + data["color"])

        x.append(data["mileage"])
        x.append(data["power"])
        x.append(data["year"])

        if self.cities_and_average_earnings == None:
            self.get_cities_and_average_earnings()

        try:
            city_index = self.cities.index(data["city"])
            average_earning = self.cities_and_average_earnings[city_index][1]
            x.append(average_earning)
        except:
            raise Exception("For column 'city', not existing value:  " + data["city"])

        return np.array([x])

    def predict(self, data):
        x = self.mapCarDataToNumericValues(data)
        x = self.scaler.transform(x)
        y_predict = self.model.predict(x)
        return y_predict

    def rmse(self, model=None):
        if model == None:
            model = self.model
        y_predict = model.predict(self.x_test)
        score = metrics.mean_squared_error(self.y_test, y_predict, squared=False)
        #If squared == True returns MSE value, if squared == False returns RMSE value.
        print("RMSE (test):", score)

    def r2(self, model=None):
        if model == None:
            model = self.model
        y_predict = model.predict(self.x_test)
        score = metrics.r2_score(self.y_test, y_predict)
        print("R2 (test):", score)

    def createLambdaMapping(self, unique_values_per_column):
        a = lambda s: unique_values_per_column.index(s)
        #a = lambda s: print(s in unique_values_per_column, s, unique_values_per_column)
        return a

    def load_data(self, data_path):
        converters = {}
        for c in self.columns_of_categorical_data:
            converters[c] = self.createLambdaMapping(self.unique_values_per_columns[c])
        converters[11] = lambda s: float(s) / 118.0 # we convert to euros

        data = np.genfromtxt(
            fname=data_path,
            dtype=float,
            delimiter=',',
            skip_header=1,
            usecols=[i for i in range(2, 12)],  # use all columns except the first (index=0)
            # missing_values=(''),
            filling_values=(-1.0),
            converters=converters,
            encoding="utf-8"
        )

        row_indexes_for_removing = []
        for r in range(len(data)):
            minus_ones = list(filter(lambda x: x == -1.0, data[r, :]))
            if len(minus_ones) > 0:
                row_indexes_for_removing.append(r)
        data = np.delete(data, row_indexes_for_removing, axis=0)  # uklanjamo redove sa nedostajucim vrednostima

        # nedostajuce vrednosti (koje su predstavljene sa -1.0) se zamenjuju sa srednjom vrednosti za odredjenu kolonu
        """
        for c in range(len(data[0])):
            without_minus_one = list(filter(lambda x: x != -1.0, data[:, c]))
            m = np.mean(without_minus_one)
            m = np.round(m)
            for i in range(len(data)):
                if data[i][c] == -1.0:
                    data[i][c] = m
        """
        return data