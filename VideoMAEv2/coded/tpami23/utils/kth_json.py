from __future__ import print_function, division
import os
import sys
import json
from glob import glob
import re
PERSON_IDS_TRAIN = [11, 12, 13, 14, 15, 16, 17, 18]
PERSON_IDS_VAL = [19, 20, 21, 23, 24, 25, 1, 4]
PERSON_IDS_TEST = [22, 2, 3, 5, 6, 7, 8, 9, 10]


def convert_csv_to_dict(video_dir_path):
    rgx = re.compile(r'^person(\d+)_.*$')
    database = {}
    for d in sorted(glob(os.path.join(video_dir_path, "*"))):
        action = os.path.basename(d)
        for path in glob(os.path.join(d, "*.avi")):
            filename, _ = os.path.splitext(os.path.basename(path))
            person_id = int(rgx.sub(r'\1', filename))
            if person_id in PERSON_IDS_TRAIN:
                subset = 'training'
            elif person_id in PERSON_IDS_VAL:
                subset = 'validation'
            elif person_id in PERSON_IDS_TEST:
                subset = 'testing'
            else:
                print(person_id)
                raise

            database[filename] = {}
            database[filename]['subset'] = subset
            database[filename]['annotations'] = {'label': action}

    return database


def get_labels(video_dir_path):
    labels = []
    for d in sorted(glob(os.path.join(video_dir_path, "*"))):
        action = os.path.basename(d)
        labels.append(action)
    return sorted(list(set(labels)))


def convert_kth_csv_to_activitynet_json(video_dir_path, dst_json_path):
    labels = get_labels(video_dir_path)
    database = convert_csv_to_dict(video_dir_path)
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    csv_dir_path = sys.argv[1]
    video_dir_path = sys.argv[2]

    dst_json_path = os.path.join(csv_dir_path, 'kth.json')
    convert_kth_csv_to_activitynet_json(video_dir_path, dst_json_path)
