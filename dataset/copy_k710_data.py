import argparse
import os
import shutil
import csv
from multiprocessing import Pool, Manager

def parse_args():
    parser = argparse.ArgumentParser(description="Copy videos from specified dataset to output directory.")
    parser.add_argument("--csv", required=True, help="Path to the input CSV file.")
    parser.add_argument("--ktype", required=True, choices=['k400', 'k600', 'k700'], help="Dataset type (k400, k600, k700).")
    parser.add_argument("--input", required=True, help="Path to the input folder (dataset root).")
    parser.add_argument("--output", required=True, help="Path to the output directory.")
    return parser.parse_args()

def find_and_copy_video(args):
    input_folder, output_folder, ktype, line = args
    video_ref = line[0].split(" ")[0]
    
    # Check if the ktype matches the dataset in the CSV entry
    dataset_type = video_ref.split("/")[0]
    if dataset_type != ktype:
        # print(f"Skipping {video_ref} as it does not match the specified dataset type {ktype}")
        return
    
    # Extract video name and construct source and destination paths
    video_name = video_ref.split("/")[-1]
    source_path = os.path.join(input_folder, video_name)
    destination_path = os.path.join(output_folder, video_name)
    
    # Copy file if it exists, otherwise log failure
    if os.path.exists(source_path):
        # check if exists
        if os.path.exists(destination_path):
            print(f"Video {video_name} already exists in output folder.")
            with open("copy_duplicated.txt", "a") as f:
                f.write(f"Existed in Target Path: {source_path}\n")
            return
        shutil.copy(source_path, destination_path)
        print(f"Copied {source_path} to {destination_path}")
    else:
        print(f"Video {video_name} not found in input folder.")
        with open("copy_failed.txt", "a") as f:
            f.write(f"{source_path}\n")

def main():
    args = parse_args()

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read CSV and prepare tasks for multiprocessing
    tasks = []
    with open(args.csv, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            tasks.append((args.input, args.output, args.ktype, line))
    # import ipdb; ipdb.set_trace()
    # Use multiprocessing to handle the copy process in parallel
    print(len(tasks))
    with Pool() as pool:
        pool.map(find_and_copy_video, tasks)

if __name__ == "__main__":
    main()
