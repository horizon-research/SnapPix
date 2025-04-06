from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool


def ffmpeg_process(args):
    file_name, dst_class_path, class_path = args
    if '.avi' not in file_name:
        return False
    if file_name.startswith('_'):
        return False
    if file_name.startswith('.'):
        return False
    name, ext = os.path.splitext(file_name)
    dst_directory_path = os.path.join(dst_class_path, name)

    video_file_path = os.path.join(class_path, file_name)
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                print('remove {}'.format(dst_directory_path))
                os.mkdir(dst_directory_path)
            else:
                return False
        else:
            os.mkdir(dst_directory_path)
    except Exception as e:
        print(e, dst_directory_path)
        return False
    cmd = 'ffmpeg -i \"{}\" -s 171x128 \"{}/image_%05d.jpg\"'.format(
        video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')


def class_process(dir_path, dst_dir_path, class_name):
    class_path = os.path.join(dir_path, class_name)
    if not os.path.isdir(class_path):
        return

    dst_class_path = os.path.join(dst_dir_path, class_name)
    if not os.path.exists(dst_class_path):
        os.mkdir(dst_class_path)

    with Pool(32) as p:
        res = p.map(ffmpeg_process, [
            (file_name, dst_class_path, class_path)
            for file_name in os.listdir(class_path)])
    return res


if __name__ == "__main__":
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    for class_name in os.listdir(dir_path):
        class_process(dir_path, dst_dir_path, class_name)
