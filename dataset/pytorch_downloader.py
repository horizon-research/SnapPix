import torchvision.datasets as datasets
import os
# Define a function to download a specific version of Kinetics dataset for both train and val splits
def download_kinetics(version, root):
    for split in ['train', 'val']:
        print(f"Downloading Kinetics-{version} {split} dataset...")
        dataset = datasets.Kinetics(
            root=os.path.join( root, f'{split}'),
            frames_per_clip=16,  # Example value, adjust as needed
            num_classes=str(version),
            split=split,
            download=True,
            num_download_workers=24  # Adjust this based on your network speed
        )
        print(f"Kinetics-{version} {split} dataset downloaded successfully!")

# Specify the root directory where the datasets will be saved
root_dir = './torch_kinetics_datasets'

# Download Kinetics-400, Kinetics-600, and Kinetics-700 (train and val)
download_kinetics(400, os.path.join(root_dir, 'kinetics400'))
download_kinetics(600, os.path.join(root_dir, 'kinetics600'))
download_kinetics(700, os.path.join(root_dir, 'kinetics700'))
