from src.data_preprocessing import get_unique_values_per_columns, merge_ads_and_average_earnings_per_cities
from src.download_data import download_ads, download_new_ads
from src.xgboost_algorithm import XGBoostAlgorithm

DOWNLOAD = False
DOWNLAOD_NEW = False
PREPROCESS = False
MERGE = False
FIT = True


# https://cloud.google.com/ml-engine/docs/training-overview
# training on google cloud AI Platform


def main():

    if DOWNLOAD:
        download_ads()

    if DOWNLAOD_NEW:
        file_path_to_old_ads = "..\\data\\ads_18-01-2020_18-00-34.csv"
        download_new_ads(file_path_to_old_ads)

    if PREPROCESS:
        data_path = "..\\data\\ads_18-01-2020_18-00-34_with_average_earnings.csv"
        get_unique_values_per_columns(data_path)

    if MERGE:
        file_path_to_ads = "..\\data\\ads_18-01-2020_18-00-34.csv"
        file_path_to_average_earnings_per_cities =  "..\\data\\prosecne-zarade-po-opstinama.csv"
        merge_ads_and_average_earnings_per_cities(file_path_to_ads, file_path_to_average_earnings_per_cities)

    if FIT:
        data_path = "..\\data\\sample.csv" #"..\\data\\ads_18-01-2020_18-00-34_with_average_earnings.csv"
        unique_values_per_columns = get_unique_values_per_columns(data_path)
        use_existing_parameters = False

        parameters = None
        if use_existing_parameters:
            parameters = {
                'rf__min_samples_split': 2, 'rf__max_features': 7, 'rf__min_samples_leaf': 1,
                'rf__criterion': 'entropy', 'rf__max_depth': 2, 'rf__n_estimators': 300,
                'gbc__min_samples_split': 2, 'gbc__max_depth': 1, 'gbc__min_samples_leaf': 3, 'gbc__subsample': 1.0,
                'gbc__max_features': 4, 'gbc__n_estimators': 350,
                # 'abc__learning_rate': 0.01, 'abc__n_estimators': 150
            }
        xgbAlgorithm = XGBoostAlgorithm(data_path, unique_values_per_columns, parameters)
        xgbAlgorithm.fit()
        #xgbAlgorithm.f1_score()


if __name__ == '__main__':
    main()