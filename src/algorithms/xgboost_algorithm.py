import sys

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


random_state = 42


class XGBoostAlgorithm:

    def __init__(self, data_path, unique_values_per_columns, existing_parameters):
        self.unique_values_per_columns = unique_values_per_columns
        self.existing_parameters = existing_parameters

        self.columns_of_categorical_data = (1,2,4,5,6,7)

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.voting_clf = None
        self.grid_search = None
        self.scaler = None

        self.initialize(data_path)

    def initialize(self, data_path):
        data = self.load_data(data_path)

        inexes_for_removing = []
        for index, row in enumerate(data):
            if row[2] >= 999999999:
            #if row[2] > 20000:
                inexes_for_removing.append(index)

        data = np.delete(data, inexes_for_removing, axis=0)

        self.y = data[:,2] # price column
        self.x = np.delete(data, 2, axis=1) # without price column

        x_len = len(self.x[0])
        for i in range(x_len):
            if i == 2:
                print("i = ", i)
                print(data[:, i])
                sns.boxplot(data=[data[:,i]])
                plt.show()
                break

        #data_dmatrix = xgb.DMatrix(data=self.X, label=self.y)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=random_state)

        self.scaler = StandardScaler()  # with_mean=False)
        # self.scaler = MinMaxScaler()
        self.regressor = XGBRegressor()

        if self.existing_parameters == None:
            kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            pipeline = Pipeline(steps=[
                ('scaler', self.scaler),
                # ('rf', RandomForestClassifier(n_estimators=100, random_state=1))
                # ('gbc', GradientBoostingClassifier(random_state=1))
                #('abc', AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='rbf'), random_state=1))
                ('xgb', self.regressor)

            ])

            parameters = {
                'xgb__objective': ['reg:squarederror'],#, 'binary:logistic'],
                'xgb__colsample_bytree': [0.0001, 0.001, 0.01, 0.3, 0.5, 1.0],
                'xgb__learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                'xgb__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'xgb__reg_alpha': [10],
                'xgb__n_estimators': [10, 50, 100]

                #'xgb__objective': ['reg:squarederror'],
                #'xgb__colsample_bytree': [0.3],
                #'xgb__learning_rate': [0.1],
                #'xgb__max_depth': [6],
                #'xgb__reg_alpha': [10],
                #'xgb__n_estimators': [10]
            }

            # parameters = {
            #     'nthread': [4],  # when use hyperthread, xgboost may become slower
            #     'objective': ['reg:squarederror'],
            #     'learning_rate': [0.05],  # so called `eta` value
            #     'max_depth': [6],
            #     'min_child_weight': [11],
            #     'silent': [1],
            #     'subsample': [0.8],
            #     'colsample_bytree': [0.7],
            #     'n_estimators': [5],  # number of trees, change it to 1000 for better results
            #     'missing': [-999],
            #     'seed': [1337]
            # }

            self.grid_search = GridSearchCV(pipeline, #self.regressor
                                            param_grid=parameters,
                                            cv=kfolds,
                                            scoring="neg_root_mean_squared_error",
                                            verbose=1,
                                            n_jobs=-1)
        else:
            pass
            # self.model = BaggingClassifier(tree.DecisionTreeClassifier(random_state=1), n_estimators=100)
            """
            self.model1 = RandomForestClassifier(
                n_estimators=self.existing_parameters['rf__n_estimators'],  # 100,
                random_state=1,
                max_features=self.existing_parameters['rf__max_features'],
                max_depth=self.existing_parameters['rf__max_depth'],
                min_samples_split=self.existing_parameters['rf__min_samples_split'],
                min_samples_leaf=self.existing_parameters['rf__min_samples_leaf'],
                criterion=self.existing_parameters['rf__criterion']
            )

            self.model2 = AdaBoostClassifier(base_estimator=SVC(probability=True, kernel='rbf'),
                                learning_rate=self.existing_parameters['abc__learning_rate'],
                               n_estimators=self.existing_parameters['abc__n_estimators']
            )
            

            self.model3 = GradientBoostingClassifier(
                n_estimators=self.existing_parameters['gbc__n_estimators'],  # 275,
                random_state=1,
                max_features=self.existing_parameters['gbc__max_features'],
                max_depth=self.existing_parameters['gbc__max_depth'],
                min_samples_split=self.existing_parameters['gbc__min_samples_split'],
                min_samples_leaf=self.existing_parameters['gbc__min_samples_leaf'],
                subsample=self.existing_parameters['gbc__subsample']
            )

            weights = [12.5, 9.5]
            self.voting_clf = VotingClassifier(estimators=[('rfc', self.model1), ('gbc', self.model3)],
                                               weights=weights, voting='soft')
            """

    def fit(self):
        if self.existing_parameters == None:
            self.grid_search.fit(self.x, self.y)
            print("Best score:", self.grid_search.best_score_)
            print("Best params:", self.grid_search.best_params_)
        else:
            self.x_train = self.scaler.fit_transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)
            # self.model.fit(self.x_train, self.y_train)
            self.voting_clf.fit(self.x_train[200:], self.y_train[200:])
            # y_predict = self.voting_clf.predict(self.x_train[:200])
            # score = metrics.f1_score(self.y_train[:200], y_predict, average='micro')
            # print("f1_score (train):", score)

    def f1_score(self):
        if self.existing_parameters == None:
            score = self.grid_search.score(self.x_test, self.y_test)
        else:
            # y_predict = self.model.predict(self.x_test)
            y_predict = self.voting_clf.predict(self.x_test)
            score = metrics.f1_score(self.y_test, y_predict, average='micro')
        # print("f1_score (test):", score)
        print(score)

    def createLambdaMapping(self, unique_values_per_column):
        a = lambda s: unique_values_per_column.index(s)
        #a = lambda s: print(s in unique_values_per_column, s, unique_values_per_column)
        return a

    def load_data(self, data_path):
        converters = {}
        for c in self.columns_of_categorical_data:
            converters[c] = self.createLambdaMapping(self.unique_values_per_columns[c])

        data = np.genfromtxt(
            fname=data_path,
            dtype=float,
            delimiter=',',
            skip_header=1,
            usecols=[i for i in range(1,12)], # use all columns except the first (index=0)
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
        data = np.delete(data, row_indexes_for_removing, axis=0) # uklanjamo redove sa nedostajucim vrednostima

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