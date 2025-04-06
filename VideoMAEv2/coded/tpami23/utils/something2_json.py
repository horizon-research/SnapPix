from __future__ import print_function, division
import os
import sys
import json


def convert_to_dict(dataset_path, labels):
    """
    database = {
      'filename': {
        'subset': 'training', # testing, validation
        'annotations': {'label': 'action'}
      }
    }
    train_json = [
      {
        "id":"78687",
        "label":"holding potato next to vicks vaporub bottle",
        "template":"Holding [something] next to [something]",
        "placeholders":["potato","vicks vaporub bottle"]
      },
    ]
    /datasets/something/somethingv2_frames_c3d/100137/000001.jpg
    """
    train_json_path = os.path.join(dataset_path, 'something-something-v2-train.json')
    test_json_path = os.path.join(dataset_path, 'something-something-v2-test.json')
    val_json_path = os.path.join(dataset_path, 'something-something-v2-validation.json')

    database = {}

    with open(train_json_path, 'r') as f:
        train_json = json.load(f)

    for d in train_json:
        filename = d['id']
        action = d['template'].replace('[', '').replace(']', '')
        if action not in labels:
            print(action)
            print(d)
            raise
        database[filename] = {}
        database[filename]['subset'] = 'training'
        database[filename]['annotations'] = {'label': action}

    with open(test_json_path, 'r') as f:
        test_json = json.load(f)

    for d in test_json:
        filename = d['id']
        action = 'NONE'
        database[filename] = {}
        database[filename]['subset'] = 'testing'
        database[filename]['annotations'] = {'label': action}

    with open(val_json_path, 'r') as f:
        val_json = json.load(f)

    for d in val_json:
        filename = d['id']
        action = d['template'].replace('[', '').replace(']', '')
        if action not in labels:
            print(action)
            print(d)
            raise
        database[filename] = {}
        database[filename]['subset'] = 'validation'
        database[filename]['annotations'] = {'label': action}

    return database


def get_labels(dataset_path):
    """
    {
      "Approaching something with your camera":"0",
    }
    """
    labels_json_path = os.path.join(dataset_path, 'something-something-v2-labels.json')
    with open(labels_json_path, 'r') as f:
        labels_json = json.load(f)
    return [action for action, i in sorted(labels_json.items(), key=lambda x: int(x[1]))]


def convert_to_activitynet_json(video_dir_path, dst_json_path):
    labels = get_labels(video_dir_path)
    database = convert_to_dict(video_dir_path, labels)
    dst_data = {}
    dst_data['labels'] = labels
    dst_data['database'] = {}
    dst_data['database'].update(database)

    with open(dst_json_path, 'w') as dst_file:
        json.dump(dst_data, dst_file)


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    dst_json_path = sys.argv[2]
    convert_to_activitynet_json(dataset_path, dst_json_path)
