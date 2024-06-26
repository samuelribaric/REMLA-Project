""" Uploads the trained model to Google Cloud Storage. """
import subprocess
from datetime import datetime

import subprocess
from datetime import datetime

def upload_model():
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_path = "models/model.keras"
    destination_path = f"gs://remla_group_9_model/model_{timestamp}.keras"
    command = f"gsutil cp {source_path} {destination_path}"
    
    # Using Python to handle cross-platform subprocess call
    subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    upload_model()
