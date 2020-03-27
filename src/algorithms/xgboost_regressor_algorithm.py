
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import warnings

from src.algorithms.algorithm_with_grid_search_cv import AlgorithmWithGridSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)


random_state = 42


class XGBoostRegressorAlgorithm(AlgorithmWithGridSearchCV):

    def __init__(self, data_path_or_data, unique_values_per_columns, existing_parameters):
        super().__init__(data_path_or_data, unique_values_per_columns, existing_parameters)

    """ Implementation of abstract method """
    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()  # with_mean=False)
        # self.scaler = MinMaxScaler()

        if self.existing_parameters == None:
            self.model = XGBRegressor(nthread=-1, random_state=random_state)

            kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            pipeline = Pipeline(steps=[
                ('scaler', self.scaler),
                ('xgb', self.model)
            ])

            """
            parameters = {
                'xgb__objective': ['reg:squarederror'],  # , 'binary:logistic'],
                'xgb__colsample_bytree': [0.0001, 0.001, 0.01, 0.3, 0.5, 1.0],
                'xgb__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
                'xgb__max_depth': [3, 4, 5, 6, 7],
                'xgb__reg_alpha': [10],
                'xgb__n_estimators': [100, 200, 300]
            }
            """

            """
            parameters = {
                'xgb__objective': ['reg:squarederror'],
                'xgb__colsample_bytree': [0.3],
                'xgb__learning_rate': [0.1],
                'xgb__max_depth': [6],
                'xgb__reg_alpha': [10],
                'xgb__n_estimators': [10]
            }
            """

            parameters = {
                'xgb__max_depth': [7],
                'xgb__colsample_bytree': [1.0],
                'xgb__reg_alpha': [10],
                'xgb__objective': ['reg:squarederror'],
                'xgb__learning_rate': [0.1],
                'xgb__n_estimators': [400]
            }

            self.grid_search = GridSearchCV(pipeline,
                                            param_grid=parameters,
                                            cv=kfolds,
                                            # scoring="neg_root_mean_squared_error",
                                            scoring="r2",
                                            verbose=1,
                                            n_jobs=-1)
        else:
            self.model = XGBRegressor(random_state=random_state,
                                       n_estimators=self.existing_parameters['xgb__n_estimators'],
                                       reg_alpha=self.existing_parameters['xgb__reg_alpha'],
                                       max_depth=self.existing_parameters['xgb__max_depth'],
                                       learning_rate=self.existing_parameters['xgb__learning_rate'],
                                       colsample_bytree=self.existing_parameters['xgb__colsample_bytree'],
                                       objective=self.existing_parameters['xgb__objective'],
                                       nthread=-1)