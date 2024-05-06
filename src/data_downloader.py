"""Utility module for downloading and setting up a dataset"""
import os
import requests
from tqdm import tqdm


def download_file(url, local_filename):
    """Downloads a file from the given URL and saves it to the specified local_filename path"""
    # Ensure the directory for the local_filename exists
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    # Stream the download
    with requests.get(url, stream=True) as request:
        request.raise_for_status()
        total_size_in_bytes = int(request.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as file:
            for data in request.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes not in (0, progress_bar.n):
            print("ERROR, something went wrong")


def setup_dataset():
    """Set up the dataset by downloading and saving necessary files (if they don't already exist)"""
    files = {
        "test.txt": "https://www.dropbox.com/scl/fi/qd31pfutruygx2ma64w1o/test.txt?rlkey=dseji8irszk99hynu18f2che9&st"
                    "=d7i4xipe&dl=1",
        "train.txt": "https://www.dropbox.com/scl/fi/jbp08vjics0gs7r1xon3u/train.txt?rlkey=iq33sint89mtsutixm96mwckz"
                     "&st=6ewkvofq&dl=1",
        "val.txt": "https://www.dropbox.com/scl/fi/uf9xiu7g1rvz3s9zelptj/val.txt?rlkey=d3pxdkn5hrr4yl7w45g7a9jh1&st"
                   "=93kugzky&dl=1"
    }
    extract_path = "data/"

    # Download each file if not already downloaded
    for file_name, url in files.items():
        if not os.path.exists(os.path.join(extract_path, file_name)):
            local_path = os.path.join(extract_path, file_name)
            download_file(url, local_path)
        else:
            print(f"File {file_name} already exists in /data, please delete existing data to download")


if __name__ == "__main__":
    setup_dataset()
