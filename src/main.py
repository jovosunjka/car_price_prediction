
from src.download_data import download_ads, download_new_ads

DOWNLOAD = True
DOWNLAOD_NEW = False

def main():
    print("Hello World")

    if DOWNLOAD:
        download_ads()

    if DOWNLAOD_NEW:
        file_path_to_old_ads = "..\\data\\ads_04-12-2019_12-06-10.csv"
        download_new_ads(file_path_to_old_ads)


if __name__ == '__main__':
    main()