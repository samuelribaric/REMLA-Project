import os
import requests
from tqdm import tqdm  

def download_file(url, local_filename):
    # Ensure the directory for the local_filename exists
    os.makedirs(os.path.dirname(local_filename), exist_ok=True)

    # Stream the download
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

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
