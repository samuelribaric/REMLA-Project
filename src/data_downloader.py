"""
This module handles the downloading of data files from remote URLs.
"""

import os
import requests

def download_file(url, local_filename):
    """Downloads a file from a URL and saves it locally."""
    if not os.path.exists(local_filename):
        print(f"Downloading {local_filename}...")
        with requests.get(url, stream=True, timeout=30) as response:
            response.raise_for_status()
            os.makedirs(os.path.dirname(local_filename), exist_ok=True)
            with open(local_filename, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        print(f"Downloaded the file: {local_filename}")
    else:
        print(f"File already exists: {local_filename}")

def setup_dataset():
    files = {
        "test.txt": "https://www.dropbox.com/scl/fi/qd31pfutruygx2ma64w1o/test.txt?rlkey=dseji8irszk99hynu18f2che9&st=d7i4xipe&dl=1",
        "train.txt": "https://www.dropbox.com/scl/fi/jbp08vjics0gs7r1xon3u/train.txt?rlkey=iq33sint89mtsutixm96mwckz&st=6ewkvofq&dl=1",
        "val.txt": "https://www.dropbox.com/scl/fi/uf9xiu7g1rvz3s9zelptj/val.txt?rlkey=d3pxdkn5hrr4yl7w45g7a9jh1&st=93kugzky&dl=1"
    }
    extract_path = "data/"

    # Download each file if not already downloaded
    for file_name, url in files.items():
        if not os.path.exists(os.path.join(extract_path, file_name)):
            local_path = os.path.join(extract_path, file_name)
            download_file(url, local_path)
        else: print(f"File {file_name} already exists in /data, please delete existing data to download")

if __name__ == "__main__":
    setup_dataset()