from __future__ import print_function, division
import os
import sys
import subprocess

def class_process(dir_path):
  print('dir_path:', dir_path)
  for file_name in os.listdir(dir_path):
    video_dir_path = os.path.join(dir_path, file_name)
    print('video_dir_path:', video_dir_path)
    image_indices = []
    for image_file_name in os.listdir(video_dir_path):
      if '.jpg' not in image_file_name:
        continue
      image_indices.append(int(image_file_name[:-4]))

    if len(image_indices) == 0:
      print('no image files', video_dir_path)
      n_frames = 0
    else:
      image_indices.sort(reverse=True)
      n_frames = image_indices[0]
      print(video_dir_path, n_frames)
    with open(os.path.join(video_dir_path, 'n_frames'), 'w') as dst_file:
      dst_file.write(str(n_frames))


if __name__=="__main__":
  dir_path = sys.argv[1]
  class_process(dir_path)

