from src.algorithms.gradient_boosting_regressor_algorithm import GradientBoostingRegressorAlgorithm
from src.data_preprocessing import get_unique_values_per_columns


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

    # bez ovog __func__ sam dobijao TypeError: 'staticmethod' object is not callable
    switcher = {
        "GRADEINT_BOOSTING_REGRESSOR": create_gradient_boosting_regressor.__func__
    }