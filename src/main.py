from src.algorithms.algorithm_factory import AlgorithmFactory
from src.data_preprocessing import get_unique_values_per_columns, merge_ads_and_average_earnings_per_cities
from src.download_data import download_ads, download_new_ads
from src.algorithms.xgboost_algorithm import XGBoostAlgorithm

from datetime import datetime


DOWNLOAD = False
DOWNLAOD_NEW = False
PREPROCESS = False
MERGE = False
FIT = True

ALGORITHM_NAME = "GRADEINT_BOOSTING_REGRESSOR"
ALGORITHM_NAME_LINEAR_REGRESION = "LINEAR_REGRESSION"


# https://cloud.google.com/ml-engine/docs/training-overview
# training on google cloud AI Platform


fuel_types = ("Benzin", "Dizel", "Benzin + Gas (TNG)", "Metan CNG", "Električni pogon", "Hibridni pogon")
fuel_type_options = [str(i+1) for i in range(len(fuel_types))]


def enter_data_about_car():
    print()
    data = {}

    new_car = "False"
    while True:
        new_car_str = input("New car ? (yes,no):  ")
        new_car_str = new_car_str.strip()
        if new_car_str in ("yes", "no"):
            if new_car_str == "yes":
                new_car = "True"
            else:
                new_car = "False"
            break
    data["new_car"] = new_car

    while True:
        brand_name = input("Enter brand name:  ")
        brand_name = brand_name.strip()
        if brand_name != "":
            break
    data["brand_name"] = brand_name

    while True:
        model_name = input("Enter model name:  ")
        model_name = model_name.strip()
        if model_name != "":
            break
    data["model_name"] = model_name

    print("\nChoose fuel type:")
    for i, fuel_type in enumerate(fuel_types):
        print("\t", str(i + 1) + ".", fuel_type)
    while True:
        fuel_type_option = input("Enter fuel type:  ")
        fuel_type_option = fuel_type_option.strip()

        if fuel_type_option in fuel_type_options:
            break
    fuel_type_index = int(fuel_type_option) - 1
    data["fuel_type"] = fuel_types[fuel_type_index]

    mileage = -1
    while True:
        mileage_str = input("\nEnter mileage (<= 500000):  ")
        mileage_str = mileage_str.strip()
        if mileage_str.isdigit():
            mileage = int(mileage_str)
            if mileage <= 500000:
                break
    data["mileage"] = mileage

    power = -1
    while True:
        power_str = input("Enter power (kW) (>= 30 and  <= 625):  ")
        power_str = power_str.strip()
        if power_str.isdigit():
            power = int(power_str)
            if power >= 30 and power <= 625:
                break
    data["power"] = power

    year = -1
    while True:
        year_str = input("Enter year (> 1995):  ")
        year_str = year_str.strip()
        if year_str.isdigit():
            year = int(year_str)
            if year > 1995 and year <= datetime.now().year:
                break
    data["year"] = year

    while True:
        color = input("Enter color:  ")
        color = color.strip()
        if color != "":
            break
    data["color"] = color

    while True:
        city = input("Enter city:  ")
        city = city.strip()
        if city != "":
            break
    data["city"] = city

    return data



def prediction(algorithm):
    data = enter_data_about_car()
    try:
        price = algorithm.predict(data)
        print("\nPrice:", price[0], "EUR")
    except Exception as e:
        print(e)

def start_cmd_app(algorithm):
    print("\nCAR PRICE PREDICTION")
    print("====================")

    while True:
        print("\n1. PREDICTION")
        print("2. EXIT")
        option = input("\nEnter option:  ")
        option = option.strip()

        if option == "1":
            prediction(algorithm)
        elif option == "2":
            break
        else:
            print("\nInvalid option!")


def main():

    if DOWNLOAD:
        download_ads()

    if DOWNLAOD_NEW:
        file_path_to_old_ads = "..\\data\\ads_12-03-2020_14-04-00.csv"
        download_new_ads(file_path_to_old_ads)

    if PREPROCESS:
        data_path = "..\\data\\ads_12-03-2020_14-04-00_with_average_earnings.csv"
        get_unique_values_per_columns(data_path)

    if MERGE:
        file_path_to_ads = "..\\data\\ads_12-03-2020_14-04-00.csv"
        file_path_to_average_earnings_per_cities =  "..\\data\\prosecne-zarade-po-opstinama.csv"
        merge_ads_and_average_earnings_per_cities(file_path_to_ads, file_path_to_average_earnings_per_cities)

    if FIT:
        """
        data_path = "..\\data\\sample.csv"
        # data_path = "..\\data\\ads_12-03-2020_14-04-00_with_average_earnings.csv"
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
        """
        #data_path = "..\\data\\sample.csv"

        #data_path = "..\\data\\ads_12-03-2020_14-04-00_with_average_earnings.csv"
        #algorithm = AlgorithmFactory.create(ALGORITHM_NAME, data_path)
        #algorithm.fit()
        #algorithm.rmse()
        #algorithm.r2()

        #start_cmd_app(algorithm)

        data_path = "..\\data\\ads_12-03-2020_14-04-00_with_average_earnings.csv"
        algorithm = AlgorithmFactory.create(ALGORITHM_NAME_LINEAR_REGRESION, data_path)
        algorithm.fit()

        algorithm.rmse()
        algorithm.r2()

if __name__ == '__main__':
    main()