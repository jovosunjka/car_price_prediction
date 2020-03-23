from src.algorithms.gradient_boosting_regressor_algorithm import GradientBoostingRegressorAlgorithm
from src.algorithms.random_forest_regressor_algorithm import RandomForestRegressorAlgorithm
from src.algorithms.voting_regressor_algorithm import VotingRegressorAlgorithm
from src.algorithms.xgboost_regressor_algorithm import XGBoostRegressorAlgorithm
from src.data_preprocessing import get_unique_values_per_columns
from src.algorithms.linear_regression_algorithm import LinearRegressionAlgorithm

class AlgorithmFactory:

    @staticmethod
    def create(algorithm_name, data_path_or_data, unique_values_per_columns=None):
        if not algorithm_name in AlgorithmFactory.switcher:
            raise NotImplementedError("Not implemented for algorithm: " + algorithm_name)
        if unique_values_per_columns == None:
            unique_values_per_columns = get_unique_values_per_columns(data_path_or_data)
        return AlgorithmFactory.switcher[algorithm_name](data_path_or_data, unique_values_per_columns)

    @staticmethod
    def create_gradient_boosting_regressor(data_path_or_data, unique_values_per_columns):
        parameters = {
                "gbr__n_estimators": 300,
                "gbr__max_features": 4,
                "gbr__max_depth": 7,
                "gbr__min_samples_leaf": 2,
                "gbr__min_samples_split": 2,
                "gbr__subsample": 1.0
        }

        #parameters = None
        return GradientBoostingRegressorAlgorithm(data_path_or_data, unique_values_per_columns, parameters)

    @staticmethod
    def create_xgboost_regressor(data_path_or_data, unique_values_per_columns):
        parameters = {
                'xgb__max_depth': 7,
                'xgb__colsample_bytree': 1.0,
                'xgb__reg_alpha': 10,
                'xgb__objective': 'reg:squarederror',
                'xgb__learning_rate': 0.1,
                'xgb__n_estimators': 400
        }
        #parameters = None
        return XGBoostRegressorAlgorithm(data_path_or_data, unique_values_per_columns, parameters)

    @staticmethod
    def create_random_forest_regressor(data_path_or_data, unique_values_per_columns):
        parameters = {
                "rf__n_estimators": 400,
                "rf__max_features": 7,
                "rf__max_depth": 15,
                "rf__min_samples_leaf": 1,
                "rf__min_samples_split": 2,
                'rf__criterion': 'mse'
        }
        #parameters = None
        return RandomForestRegressorAlgorithm(data_path_or_data, unique_values_per_columns, parameters)

    @staticmethod
    def create_voting_regressor(data_path_or_data, unique_values_per_columns):
        parameters = {
            'vr__weights': [0, 3, 2]
        }
        #parameters = None
        return VotingRegressorAlgorithm(data_path_or_data, unique_values_per_columns, parameters)

    @staticmethod
    def create_linear_regression(data_path_or_data, unique_values_per_columns):
        return LinearRegressionAlgorithm(data_path_or_data, unique_values_per_columns)

    @staticmethod
    def get_algorithm_names():
        algorithm_names = list(AlgorithmFactory.switcher.keys())
        algorithm_names.sort()
        return algorithm_names

    # bez ovog __func__ sam dobijao TypeError: 'staticmethod' object is not callable
    switcher = {
        "GRADEINT_BOOSTING_REGRESSOR": create_gradient_boosting_regressor.__func__,
        "XGBOOST_REGRESSOR": create_xgboost_regressor.__func__,
        "RANDOM_FOREST_REGRESSOR": create_random_forest_regressor.__func__,
        "VOTING_REGRESSOR": create_voting_regressor.__func__,
        "LINEAR_REGRESSION": create_linear_regression.__func__
    }