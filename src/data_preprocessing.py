import csv


columns = ["AdID", "city", "new", "price", "brandName", "modelName", "fuelType", "color", "mileage", "power", "year"]


def get_unique_values_per_columns(file_path_to_ads):

    rows = []

    with open(file_path_to_ads, "r", newline="", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # preskoci header
        for row in reader:
           rows.append(row)

    column_indexes = (1,2,4,5,6,7)
    unique_values_per_columns = {}

    column_values = list(zip(*rows))

    for i in column_indexes:
            tmp_list = list(filter(lambda x: x != "", list(set(column_values[i]))))
            tmp_list.sort()
            unique_values_per_columns[i] = tmp_list
            print(unique_values_per_columns[i])

    return unique_values_per_columns

def get_cities_and_average_earnings(file_path_to_average_earnings_per_cities):
    cities_and_average_earnings = []

    with open(file_path_to_average_earnings_per_cities, "r", newline="", encoding='iso-8859-2') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # preskoci header
        for row_index, row in enumerate(reader):
            cities_and_average_earnings.append(row)
        return cities_and_average_earnings

def merge_ads_and_average_earnings_per_cities(file_path_to_ads, file_path_to_average_earnings_per_cities):
    cities_and_average_earnings = get_cities_and_average_earnings(file_path_to_average_earnings_per_cities)

    cities = list(map(lambda x: x[0], cities_and_average_earnings))

    len_columns = len(columns)
    ads = {}

    with open(file_path_to_ads, "r", newline="", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # preskoci header
        for row_index, row in enumerate(reader):
            if len(row) != len_columns:
                raise Exception("The number of columns in the file (" + file_path_to_ads + ") in row " +
                                (row_index + 1) + ". must be " + len_columns + ".")
            ad_id = row[0]
            ads[ad_id] = {columns[i]: row[i] for i in range(len_columns)}
            city = row[1]
            try:
                index = cities.index(city)
                average_earning = cities_and_average_earnings[index][1]
                ads[ad_id]["average_earning"] = average_earning
            except:
                del ads[ad_id]

    columns_with_average_earning = []
    columns_with_average_earning.extend(columns)
    columns_with_average_earning.append("average_earning")

    file_path_to_ads_with_average_earnings = file_path_to_ads.replace(".csv", "") + "_with_average_earnings.csv"
    with open(file_path_to_ads_with_average_earnings, "w", newline="", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns_with_average_earning)
        for ad in ads.values():
            ad_values = [ad[col] for col in columns_with_average_earning]
            print(ad_values)
            writer.writerow(ad_values)

