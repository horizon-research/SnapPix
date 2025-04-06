from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "WK1997/MAEv2"

# List of folder paths
folder_paths = [
    "mmdataset/k710_train_tar",
    "mmdataset/ssv2_tar",
    "mmdataset/ucf_tar"
]

# Loop through each folder and then each file within the folder
for folder_path in folder_paths:
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):  # Ensure it's a file
            print(f"Uploading {file_name} from {folder_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.join(file_name),
                repo_id=repo_id,
                repo_type="dataset",
            )