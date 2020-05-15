import requests
import csv
from datetime import datetime
import time



GET_ADS_BASE_URL = "https://www.polovniautomobili.com/json/v3/getAds"
# koristimo v3 zato sto nudi sve potrebne kolone i zato sto nudi izlistavanje svih page-ova
GET_AD_DETAILS_BASE_URL = "https://www.polovniautomobili.com/json/v3/getAdDetails/"

HEADERS = { "accept": "application/json; charset=utf-8" }

columns = ["AdID", "city", "new", "price", "brandName", "modelName", "fuelType", "color", "mileage", "power", "year", "engineVolume", "gearBox", "emissionClass"]

NUMBER_OF_PAGES = 3000

NUMBER_OF_ATTEMPTS = 3

def get_response(url):
    counter = 0
    while counter < NUMBER_OF_ATTEMPTS:  # pokusaj nekoliko puta
        try:
            response = requests.get(url=url, headers=HEADERS)
            if response.status_code == 200:
                break
        except Exception as e:
            print(e)

        counter += 1
    if counter == NUMBER_OF_ATTEMPTS:
        message = url + " failed " + str(NUMBER_OF_ATTEMPTS) + " times"
        with open("..\\data\\errors.txt", "a") as errors_file:
            errors_file.write(message)
        raise Exception(message)

    response_json = response.json()
    #print(response_json)
    return response_json

def download_ads_with_details(ads={}):
    start_time = time.time()

    print("Downloading ads with details...")
    #i = 0
    #while True:

    for i in range(NUMBER_OF_PAGES):
        url = GET_ADS_BASE_URL + "?SortingType=1&category=26&pageID=" + str(i)
        response_json = None
        try:
            response_json = get_response(url)
        except Exception as e:
            print()
            print(e)
            continue

        #classifieds = r_json["payloadData"]["classifieds"]
        classifieds = response_json["classifieds"]

        index = 1

        print("=============\n")
        print("Classifieds:\n")
        for classified in classifieds:
            ad_id = str(classified["AdID"])

            if not ad_id in ads:
                url = GET_AD_DETAILS_BASE_URL + ad_id
                response_json = None
                try:
                    response_json = get_response(url)
                except Exception as e:
                    print()
                    print(e)
                    continue

                print(str(index) + ".  id = " + ad_id)

                data_needed = {}
                data_needed["engineVolume"] = response_json["engineVolume"] if "engineVolume" in response_json else None
                data_needed["gearBox"] = response_json["gearBox"] if "gearBox" in response_json else None
                data_needed["emissionClass"]= response_json["emissionClass"] if "emissionClass" in response_json else None

                for col in columns:
                    if col in classified:
                        data_needed[col] = classified[col]
                    else:
                        data_needed[col] = None
                ads[ad_id] = data_needed
            else:
                print(str(index) + ".  (DUPLICATE) id = " + ad_id)
            index += 1
        #i += 1
        print("page_" + str(i) + " done | len(ads)=="+str(len(ads.keys())))
    print("Downloading ads with details...done\n")

    now = datetime.now()
    file_name = "ads_with_details_" + now.strftime("%d-%m-%Y_%H-%M-%S") + ".csv"
    file_path = "..\\data\\" + file_name
    with open(file_path, "w", newline="", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for ad in ads.values():
            ad_values = [ ad[col] for col in columns ]
            print(ad_values)
            writer.writerow(ad_values)

    print(ads)
    print("len(ads.keys()) == " + str(len(ads.keys())))


    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_min = int(elapsed_time // 60) # celobrojno deljenje
    elapsed_time_sec = round(elapsed_time % 60)
    print("Elapsed time: " + str(elapsed_time_min) + " min " + str(elapsed_time_sec) + " sec")

def download_ads(ads={}):
    start_time = time.time()

    print("Downloading ads...")
    #i = 0
    #while True:

    for i in range(NUMBER_OF_PAGES):
        url = GET_ADS_BASE_URL + "?SortingType=1&category=26&pageID=" + str(i)
        response_json = None
        try:
            response_json = get_response(url)
        except Exception as e:
            print()
            print(e)
            continue

        #classifieds = r_json["payloadData"]["classifieds"]
        classifieds = response_json["classifieds"]
        index = 1

        print("=============\n")
        print("Classifieds:\n")
        for classified in classifieds:
            ad_id = str(classified["AdID"])

            if not ad_id in ads:
                print(str(index) + ".  id = " + ad_id)
                data_needed = {}
                for col in columns:
                    if col in classified:
                        data_needed[col] = classified[col]
                    else:
                        data_needed[col] = None
                    ads[ad_id] = data_needed
            else:
                print(str(index) + ".  (DUPLICATE) id = " + ad_id)
            index += 1
        #i += 1
        print("page_" + str(i) + " done | len(ads)=="+str(len(ads.keys())))
    print("Downloading ads...done\n")

    now = datetime.now()
    file_name = "ads_" + now.strftime("%d-%m-%Y_%H-%M-%S") + ".csv"
    file_path = "..\\data\\" + file_name
    with open(file_path, "w", newline="", encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for ad in ads.values():
            ad_values = [ ad[col] for col in columns ]
            print(ad_values)
            writer.writerow(ad_values)

    print(ads)
    print("len(ads.keys()) == " + str(len(ads.keys())))


    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time_min = int(elapsed_time // 60) # celobrojno deljenje
    elapsed_time_sec = round(elapsed_time % 60)
    print("Elapsed time: " + str(elapsed_time_min) + " min " + str(elapsed_time_sec) + " sec")

def download_new_ads(file_path_to_old_ads):
    """
    Ova metoda ce ucitati stare oglase iz csv fajla, dodace samo nove i sve to ce upisati u novi fajl
    """

    ads = get_ads(file_path_to_old_ads)
    download_ads(ads)

def get_ads(file_path_to_old_ads):
    len_columns = len(columns)
    ads = {}

    with open(file_path_to_old_ads, "r", newline="", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # preskoci header
        for row_index, row in enumerate(reader):
            if len(row) != len_columns:
                raise Exception("The number of columns in the file (" + file_path_to_old_ads + ") in row " +
                                (row_index + 1) + ". must be " + len_columns + ".")
            ad_id = row[0]
            ads[ad_id] = {columns[i]: row[i] for i in range(len_columns)}
    return ads

def get_ad_ids(file_path_to_old_ads):
    len_columns = len(columns)
    ad_ids = []

    with open(file_path_to_old_ads, "r", newline="", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        next(reader)  # preskoci header
        for row_index, row in enumerate(reader):
            if len(row) != len_columns:
                raise Exception("The number of columns in the file (" + file_path_to_old_ads + ") in row " +
                                (row_index + 1) + ". must be " + len_columns + ".")
            ad_id = row[0]
            ad_ids.append(ad_id)
    return ad_ids