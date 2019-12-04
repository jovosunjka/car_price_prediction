import requests
import csv
from datetime import datetime
import time



BASE_URL = "https://www.polovniautomobili.com/json/v3/getAds"
# koristimo v3 zato sto nudi sve potrebne kolone i zato sto nudi izlistavanje svih page-ova

HEADERS = { "accept": "application/json; charset=utf-8" }

columns = ["AdID", "city", "new", "price", "brandName", "modelName", "fuelType", "color", "mileage", "power", "year"]

NUMBER_OF_PAGES = 3000

NUMBER_OF_ATTEMPTS = 3


def download_ads(ads={}):
    start_time = time.time()

    print("Downloading ads...")
    #i = 0
    #while True:
    for i in range(NUMBER_OF_PAGES):
        counter = 0;
        while counter < NUMBER_OF_ATTEMPTS: # pokusaj nekoliko puta
            r = requests.get(url=BASE_URL + "?SortingType=1&category=26&pageID="+ str(i), headers=HEADERS)
            if r.status_code == 200:
                break;
            counter += 1
        if counter == NUMBER_OF_ATTEMPTS:
            message = "failed " + str(NUMBER_OF_ATTEMPTS) + " times (pageID = " + str(i) + ")\n"
            with open("..\\data\\errors.txt", "a") as errors_file:
                errors_file.write(message)
            print(message)
            continue;

        r_json = r.json()
        print(r_json)

        #classifieds = r_json["payloadData"]["classifieds"]
        classifieds = r_json["classifieds"]
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

    len_columns = len(columns)
    ads = {}

    with open(file_path_to_old_ads, "r", newline="", encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="|", quoting=csv.QUOTE_MINIMAL)
        next(reader) # preskoci header
        for row_index, row in enumerate(reader):
            if len(row) != len_columns:
                raise Exception("The number of columns in the file (" + file_path_to_old_ads + ") in row " +
                                (row_index+1) +". must be " + len_columns + ".")
            ad_id = row[0]
            ads[ad_id] = {columns[i]: row[i] for i in range(len_columns)}


    download_ads(ads)

