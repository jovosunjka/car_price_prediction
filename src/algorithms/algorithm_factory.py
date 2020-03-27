from src.algorithms.gradient_boosting_regressor_algorithm import GradientBoostingRegressorAlgorithm
from src.algorithms.xgboost_regressor_algorithm import XGBoostRegressorAlgorithm
from src.data_preprocessing import get_unique_values_per_columns
from src.algorithms.linear_regression_algorithm import LinearRegressionAlgorithm
from src.algorithms.polynomial_regression_algorithm import PolynomialRegressionAlgorithm
from src.algorithms.knn_algorithm import KnnAlgorithm

class AlgorithmFactory:

    @staticmethod
    def create(algorithm_name, data_path):
        if not algorithm_name in AlgorithmFactory.switcher:
            raise NotImplementedError("Not implemented for algorithm: " + algorithm_name)
        unique_values_per_columns = get_unique_values_per_columns(data_path)
        return AlgorithmFactory.switcher[algorithm_name](data_path, unique_values_per_columns)

    @staticmethod
    def create_gradient_boosting_regressor(data_path, unique_values_per_columns):
        parameters = {
                "gbr__n_estimators": 300,
                "gbr__max_features": 4,
                "gbr__max_depth": 7,
                "gbr__min_samples_leaf": 2,
                "gbr__min_samples_split": 2,
                "gbr__subsample": 1.0
        }

        #parameters = None
        return GradientBoostingRegressorAlgorithm(data_path, unique_values_per_columns, parameters)

    @staticmethod
    def create_xgboost_regressor(data_path, unique_values_per_columns):
        parameters = {
            'xgb__objective': 'reg:squarederror',
            'xgb__colsample_bytree': 0.3,
            'xgb__learning_rate': 0.1,
            'xgb__max_depth': 6,
            'xgb__reg_alpha': 10,
            'xgb__n_estimators': 10
        }
        # parameters = None
        return XGBoostRegressorAlgorithm(data_path, unique_values_per_columns, parameters)

    @staticmethod
    def create_linear_regression(data_path, unique_values_per_columns):
        return LinearRegressionAlgorithm(data_path, unique_values_per_columns)

    @staticmethod
    def create_polynomial_regression(data_path, unique_values_per_columns):
        return PolynomialRegressionAlgorithm(data_path, unique_values_per_columns)

    @staticmethod
    def create_knn(data_path, unique_values_per_columns):
        parameters = {
            'n_neighbors': 5,
            'weights' : 'distance',
            'p' : 1
        }
        return KnnAlgorithm(data_path, unique_values_per_columns, parameters)

    @staticmethod
    def get_algorithm_names():
        return list(AlgorithmFactory.switcher.keys())

    # bez ovog __func__ sam dobijao TypeError: 'staticmethod' object is not callable
    switcher = {
        "GRADEINT_BOOSTING_REGRESSOR": create_gradient_boosting_regressor.__func__,
        "XGBOOST_REGRESSOR": create_xgboost_regressor.__func__,
        "LINEAR_REGRESSION": create_linear_regression.__func__,
        "POLYNOMIAL_REGRESSION": create_polynomial_regression.__func__,
        "KNN": create_knn.__func__
    }