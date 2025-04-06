import argparse
import os
import csv

def main():
    parser = argparse.ArgumentParser(description="Process and check video files.")
    parser.add_argument("k710_csv", help="Path to k710train.csv")
    parser.add_argument("ssv2_csv", help="Path to ssv2train.csv")
    parser.add_argument("k400_folder", help="Path to k400_preprocessed_folder")
    parser.add_argument("k600_folder", help="Path to k600_preprocessed_folder")
    parser.add_argument("k700_folder", help="Path to k700_preprocessed_folder")
    parser.add_argument("ssv2_folder", help="Path to SSV2_preprocessed_folder")
    parser.add_argument("output_folder", help="Path to output_folder")
    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)
    # Output paths
    combined_output_path = os.path.join(args.output_folder, "pretrained_combine.csv")
    failed_log_path = os.path.join(args.output_folder, "gen_list_failed.log")

    # clean the output files by rm -rf output_folder
    if os.path.exists(combined_output_path):
        os.remove(combined_output_path)

    # Initialize outputs
    combined_rows = []
    failed_log = []

    # Process k710train.csv
    with open(args.k710_csv, "r") as k710_file:
        k710_reader = csv.reader(k710_file, delimiter=" ")
        for row in k710_reader:
            path, _ = row[0], row[1]
            dataset = path.split('/')[0]
            video_name = os.path.basename(path)
            
            if dataset == "k400":
                folder = args.k400_folder
            elif dataset == "k600":
                folder = args.k600_folder
            elif dataset == "k700":
                folder = args.k700_folder
            else:
                failed_log.append("Unknown dataset: " + row[0])
                continue

            # Check if the preprocessed video file exists
            if os.path.exists(os.path.join(folder, f"{video_name}")):
                combined_rows.append(f"k710_train/{video_name} 0 -1")
            else:
                failed_log.append(row[0])

    # Process ssv2train.csv
    with open(args.ssv2_csv, "r") as ssv2_file:
        ssv2_reader = csv.reader(ssv2_file, delimiter=" ")
        for row in ssv2_reader:
            path, *_ = row[0], row[1:]
            video_name = os.path.basename(path)
            # import ipdb; ipdb.set_trace()
            if os.path.exists(os.path.join(args.ssv2_folder, f"{video_name}.mp4")):
                combined_rows.append(f"ssv2_processed/{video_name}.mp4 0 -1")
            else:
                failed_log.append(row[0])

    # Write the combined output file
    with open(combined_output_path, "w") as output_file:
        for line in combined_rows:
            output_file.write(line + "\n")

    # Write the failed log file
    with open(failed_log_path, "w") as log_file:
        for line in failed_log:
            log_file.write(line + "\n")

if __name__ == "__main__":
    main()
