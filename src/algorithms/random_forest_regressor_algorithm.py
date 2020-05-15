
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.algorithm_with_grid_search_cv import AlgorithmWithGridSearchCV

random_state = 42



class RandomForestRegressorAlgorithm(AlgorithmWithGridSearchCV):

    def __init__(self, data_path_or_data, unique_values_per_columns, existing_parameters):
        super().__init__(data_path_or_data, unique_values_per_columns, existing_parameters)

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()  # with_mean=False)
        # self.scaler = MinMaxScaler()

        if self.existing_parameters == None:
            self.model = RandomForestRegressor(random_state=random_state)

            kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            pipeline = Pipeline(steps=[
                ('scaler', self.scaler),
                ('rf', self.model)
            ])

            """
            parameters = {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [4, 5, 6, 7],
                'rf__max_features': [3, 4, 5, 6, 7],
                'rf__min_samples_leaf': [1, 2, 3],
                'rf__min_samples_split': [2, 3, 4],
                'rf__criterion': ['mse', 'mae', 'gini', 'entropy']
            }
            """

            parameters = {
                "rf__n_estimators": [400],
                "rf__max_features": [7],
                "rf__max_depth": [15],
                "rf__min_samples_leaf": [1],
                "rf__min_samples_split": [2],
                'rf__criterion': ['mse']
            }

            self.grid_search = GridSearchCV(pipeline,
                                            param_grid=parameters,
                                            cv=kfolds,
                                            #scoring="neg_root_mean_squared_error",
                                            scoring="r2",
                                            verbose=1,
                                            n_jobs=-1)
        else:
            self.model = RandomForestRegressor(random_state=random_state,
                                            n_estimators=self.existing_parameters['rf__n_estimators'],  # 275,
                                            max_features=self.existing_parameters['rf__max_features'],
                                            max_depth=self.existing_parameters['rf__max_depth'],
                                            min_samples_split=self.existing_parameters['rf__min_samples_split'],
                                            min_samples_leaf=self.existing_parameters['rf__min_samples_leaf'],
                                            criterion=self.existing_parameters['rf__criterion']
            )