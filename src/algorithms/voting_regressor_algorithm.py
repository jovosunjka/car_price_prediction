
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.algorithm_with_grid_search_cv import AlgorithmWithGridSearchCV

random_state = 42



class VotingRegressorAlgorithm(AlgorithmWithGridSearchCV):

    def __init__(self, data_path_or_data, unique_values_per_columns, existing_parameters):
        super().__init__(data_path_or_data, unique_values_per_columns, existing_parameters)

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()  # with_mean=False)
        # self.scaler = MinMaxScaler()

        data = {
            "x_train": self.x_train,
            "x_test": self.x_test,
            "y_train": self.y_train,
            "y_test": self.y_test
        }

        from src.algorithms.algorithm_factory import AlgorithmFactory
        pr = AlgorithmFactory.create("POLYNOMIAL_REGRESSION", data, self.unique_values_per_columns)
        gbr = AlgorithmFactory.create("GRADEINT_BOOSTING_REGRESSOR", data, self.unique_values_per_columns)
        xgb = AlgorithmFactory.create("XGBOOST_REGRESSOR", data, self.unique_values_per_columns)
        rf = AlgorithmFactory.create("RANDOM_FOREST_REGRESSOR", data, self.unique_values_per_columns)

        estimators = [('pr', pr.model), ('gbr', gbr.model), ('xgb', xgb.model), ('rf', rf.model)]

        if self.existing_parameters == None:
            self.model = VotingRegressor(estimators=estimators)

            kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            pipeline = Pipeline(steps=[
                ('scaler', self.scaler),
                ('vr', self.model)
            ])

            parameters = {
                'vr__weights': [
                    [0, 3, 0, 0],
                    [0, 2, 1, 0],
                    [1, 10, 8, 1],
                    [0, 5, 3, 1],
                    [1, 8, 6, 2],
                    [1, 4, 3, 2]

                ]
            }

            """
            parameters = {
                'vr__weights': [
                    [0, 3, 2]
                ]
            }
            """

            self.grid_search = GridSearchCV(pipeline,
                                            param_grid=parameters,
                                            cv=kfolds,
                                            #scoring="neg_root_mean_squared_error",
                                            scoring="r2",
                                            verbose=1,
                                            n_jobs=-1)
        else:
            self.model = VotingRegressor(estimators=estimators,
                                            weights=self.existing_parameters['vr__weights'])