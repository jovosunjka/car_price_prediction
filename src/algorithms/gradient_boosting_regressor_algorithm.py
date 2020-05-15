
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.algorithm_with_grid_search_cv import AlgorithmWithGridSearchCV

random_state = 42



class GradientBoostingRegressorAlgorithm(AlgorithmWithGridSearchCV):

    def __init__(self, data_path_or_data, unique_values_per_columns, existing_parameters):
        super().__init__(data_path_or_data, unique_values_per_columns, existing_parameters)

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()  # with_mean=False)
        # self.scaler = MinMaxScaler()

        if self.existing_parameters == None:
            self.model = GradientBoostingRegressor(random_state=random_state)

            kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            pipeline = Pipeline(steps=[
                ('scaler', self.scaler),
                ('gbr', self.model)
            ])


            parameters = {
                'gbr__n_estimators': [100, 200, 300],
                #'gbr__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'gbr__max_depth': [4, 5, 6, 7],
                #'gbr__max_features': [1, 2, 3, 4, 5, 6, 7],
                'gbr__max_features': [3, 4, 5, 6, 7],
                #'gbr__min_samples_leaf': [1, 2, 3, 4, 5],
                'gbr__min_samples_leaf': [1, 2, 3],
                #'gbr__min_samples_split': [2, 3, 4, 5],
                'gbr__min_samples_split': [2, 3, 4],
                #'gbr__subsample': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
                'gbr__subsample': [0.001, 0.01, 0.1, 1.0]
            }

            """
            parameters = {
                "gbr__n_estimators": [400],
                "gbr__max_depth": [8],
                "gbr__max_features": [5],
                "gbr__min_samples_leaf": [3],
                "gbr__min_samples_split": [2],
                "gbr__subsample": [1.0]
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
            self.model = GradientBoostingRegressor(random_state=random_state,
                                            n_estimators=self.existing_parameters['gbr__n_estimators'],  # 275,
                                            max_features=self.existing_parameters['gbr__max_features'],
                                            max_depth=self.existing_parameters['gbr__max_depth'],
                                            min_samples_split=self.existing_parameters['gbr__min_samples_split'],
                                            min_samples_leaf=self.existing_parameters['gbr__min_samples_leaf'],
                                            subsample=self.existing_parameters['gbr__subsample']
            )

    def fit(self):
        self.x_train = self.pca.fit_transform(self.x_train)
        self.x_test = self.pca.transform(self.x_test)