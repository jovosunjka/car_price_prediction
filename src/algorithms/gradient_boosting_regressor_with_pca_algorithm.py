from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.algorithms.algorithm_with_grid_search_cv import AlgorithmWithGridSearchCV

random_state = 42



class GradientBoostingRegressorWithPcaAlgorithm(AlgorithmWithGridSearchCV):

    def __init__(self, data_path_or_data, unique_values_per_columns, existing_parameters):
        super().__init__(data_path_or_data, unique_values_per_columns, existing_parameters)
        self.pca = None

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()  # with_mean=False)
        # self.scaler = MinMaxScaler()

        if self.existing_parameters == None:
            self.model = GradientBoostingRegressor(random_state=random_state)

            kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            pipeline = Pipeline(steps=[
                ('scaler', self.scaler),
                ('pca', PCA(random_state=1)),
                ('gbr', self.model)
            ])


            parameters = {
                'gbr__n_estimators': [200, 300, 400],
                #'gbr__max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                'gbr__max_depth': [4, 5, 6, 7, 8],
                #'gbr__max_features': [1, 2, 3, 4, 5, 6, 7],
                #'gbr__max_features': [3, 4, 5, 6, 7],
                #'gbr__min_samples_leaf': [1, 2, 3, 4, 5],
                'gbr__min_samples_leaf': [1, 2, 3],
                #'gbr__min_samples_split': [2, 3, 4, 5],
                'gbr__min_samples_split': [2, 3, 4],
                #'gbr__subsample': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]
                'gbr__subsample': [0.1, 1.0],

                'pca__n_components': [6, 7, 8],
                'pca__svd_solver': ['auto', 'full', 'randomized'],
                'pca__whiten': [False]
            }

            """
            parameters = {
                "gbr__n_estimators": [400],
                "gbr__max_depth": [8],
                "gbr__max_features": [5],
                "gbr__min_samples_leaf": [3],
                "gbr__min_samples_split": [2],
                "gbr__subsample": [1.0],

                'pca__n_components': [1, 2, 3, 4, 5, 6, 7, 8],
                'pca__svd_solver': ['auto', 'full', 'arpack', 'randomized'],
                'pca__whiten': [False, True]
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

            self.pca = PCA(
                random_state=1,
                n_components=self.existing_parameters['pca__n_components'],
                whiten=self.existing_parameters['pca__whiten'],
                svd_solver=self.existing_parameters['pca__svd_solver']
            )

    """ Override """
    def fit(self):
        if self.existing_parameters == None:
            self.grid_search.fit(self.x_train, self.y_train)
            print("Best score:", self.grid_search.best_score_)
            print("Best params:", self.grid_search.best_params_)
        else:
            self.x_train = self.scaler.fit_transform(self.x_train)
            self.x_test = self.scaler.transform(self.x_test)
            self.x_train = self.pca.fit_transform(self.x_train)
            self.x_test = self.pca.transform(self.x_test)
            self.model.fit(self.x_train, self.y_train)

    """ Override """
    def predict(self, data):
        x = self.mapCarDataToNumericValues(data)
        x = self.scaler.transform(x)
        x = self.pca.transform(x)
        y_predict = self.model.predict(x)
        return y_predict