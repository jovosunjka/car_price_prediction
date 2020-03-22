import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from datetime import datetime

random_state = 42

class LinearRegressionAlgorithm:
    def __init__(self, data_path, unique_values_per_columns):
        self.unique_values_per_columns = unique_values_per_columns

        self.columns_of_categorical_data = (2, 4, 5, 6, 7)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.scaler = None

        self.model = None

        self.initialize(data_path)

    def initialize(self, data_path):
        data = self.load_data(data_path)
        data = self.remove_rows_with_big_values(data)

        y = data[:, 1]  # price column
        x = np.delete(data, 1, axis=1)  # without price column

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2,
                                                                                random_state=random_state)
        self.scaler = StandardScaler()
        self.model = LinearRegression()

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

    def load_data(self, data_path):
        converters = {}
        for c in self.columns_of_categorical_data:
            converters[c] = self.createLambdaMapping(self.unique_values_per_columns[c])
        converters[11] = lambda s: float(s) / 118.0 # pretvaramo u evre

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

    def createLambdaMapping(self, unique_values_per_column):
        a = lambda s: unique_values_per_column.index(s)
        #a = lambda s: print(s in unique_values_per_column, s, unique_values_per_column)
        return a

    def fit(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)
        self.model.fit(self.x_train, self.y_train)

    def rmse(self):
        y_predict = self.model.predict(self.x_test)
        score = metrics.mean_squared_error(self.y_test, y_predict, squared=False)
        print("RMSE linear regression(test):", score)

    def r2(self):
        y_predict = self.model.predict(self.x_test)
        score = metrics.r2_score(self.y_test, y_predict)
        print("R2 linear regression(test):", score)
