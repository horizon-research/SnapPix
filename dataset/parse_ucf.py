import csv
import os
from glob import glob

# Path to the directory containing the files
dataset_path = "/workspace/CodedExposure/dataset/mmdataset/OpenDataLab___UCF101/raw/UCF101/source_compressed/data/ucfTrainTestlist"

# Parse classInd.txt to create a dictionary for label mapping
class_ind_path = os.path.join(dataset_path, "classInd.txt")
class_dict = {}
with open(class_ind_path, "r") as f:
    for line in f:
        class_id, class_name = line.strip().split()
        class_dict[class_name] = int(class_id)

# Functions to generate CSV rows for train, test, and pretrain
def parse_trainlist(file_path):
    pretrain_rows = []
    train_rows = []
    errors = []

    with open(file_path, "r") as f:
        for line in f:
            path_label = line.strip().split()
            video_path = path_label[0].replace(".avi", ".mp4")
            actual_label = int(path_label[1]) - 1

            # Extract class name and lookup label
            class_name = video_path.split("/")[0]
            video_name = video_path.split("/")[-1]
            if class_name in class_dict:
                label = int(class_dict[class_name]) - 1
                if label != actual_label:
                    errors.append(f"Error in {video_path}: Expected label {label}, found {actual_label}")
                train_rows.append([f"ucf/{video_name} {label}"])

    return train_rows, errors

def parse_testlist(file_path):
    test_rows = []
    with open(file_path, "r") as f:
        for line in f:
            video_path = line.strip().replace(".avi", ".mp4")
            class_name = video_path.split("/")[0]
            video_name = video_path.split("/")[-1]
            if class_name in class_dict:
                # check if class name is in class dictionary
                if class_name in class_dict:
                    label = int(class_dict[class_name]) - 1
                else:
                    with open("ucf_parse_err.log", "a") as err_log:
                        err_log.write(f"Error in {video_path}: Class name not found in class dictionary\n")
                test_rows.append([f"ucf/{video_name} {label}"])
    return test_rows

# Creating output directory and clearing old files
output_dir = "ucf101_list"
os.makedirs(output_dir, exist_ok=True)
os.system(f"rm -rf {output_dir}/*")

# Parsing all trainlist and testlist files, generating separate CSV files for each
errors = []

for trainlist_file in glob(os.path.join(dataset_path, "trainlist*.txt")):
    filename = os.path.splitext(os.path.basename(trainlist_file))[0]
    train_rows, errs = parse_trainlist(trainlist_file)
    errors.extend(errs)

    with open(f"{output_dir}/{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(train_rows)

for testlist_file in glob(os.path.join(dataset_path, "testlist*.txt")):
    filename = os.path.splitext(os.path.basename(testlist_file))[0]
    test_rows = parse_testlist(testlist_file)

    # Writing test CSV file for this specific list
    with open(f"{output_dir}/{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(test_rows)

# Logging errors if any
if errors:
    with open("ucf_parse_err.log", "w") as f:
        f.write("\n".join(errors))

print(f"Generated files with {len(errors)} labeling errors.")
