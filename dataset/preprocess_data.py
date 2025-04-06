import os
import cv2
import argparse
from multiprocessing import Pool

import numpy as np

def inverse_gamma_table():
    x = np.linspace(0, 1, 256)  # 0~255
    table = np.where(
        x <= 0.04045,
        (x / 12.92) * 255,  
        ((x + 0.055) / 1.055) ** 2.4 * 255  
    )
    # Clip values to ensure they are between 0 and 255
    table = np.clip(table, 0, 255)

    return table.astype(np.uint8)


def process_video(input_path, output_folder):
    try:
        # Read the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video {input_path}")
            with open('ds_failed.txt', 'a') as f:
                f.write(input_path + '\n')
            return
        
        # Get the video file name
        video_name = os.path.basename(input_path)
        video_name = os.path.splitext(video_name)[0] + '.mp4'
        output_path = os.path.join(output_folder, video_name)

        # Define video writer with the same fps, frame size will be determined later
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Read the first frame to determine the resizing
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read frames from {input_path}")
            cap.release()
            with open('preprocess_failed.txt', 'a') as f:
                f.write(input_path + '\n')
            return

        # Resize keeping the short side to 112 and applying inverse gamma correction
        height, width = frame.shape[:2]
        scale_factor = 112 / min(height, width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        output_frame_size = (new_width, new_height)

        # Check if the file already exists
        if os.path.exists(output_path):
            print(f"File already exists: {output_path}")
            with open('preprocess_duplicate.txt', 'a') as f:
                f.write(input_path + '\n')
            cap.release()
            return

        # Create the video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, output_frame_size, isColor=False)

        # Precompute the inverse gamma correction lookup table
        table = inverse_gamma_table()


        # Process frames
        while ret:
            # Apply inverse gamma correction
            frame = cv2.LUT(frame, table)

            # Resize and convert to grayscale
            gray_frame = cv2.cvtColor(cv2.resize(frame, output_frame_size), cv2.COLOR_BGR2GRAY)
            out.write(gray_frame)

            ret, frame = cap.read()

        # Release resources
        cap.release()
        out.release()
        print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def process_videos_in_folder(source_folder, output_folder, input_format):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    video_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(source_folder)
        for file in files if file.endswith(input_format)
    ]

    return [(video, output_folder) for video in video_files]

def main():
    parser = argparse.ArgumentParser(description='Process and resize videos with inverse gamma correction.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder')
    parser.add_argument('output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--input_format', type=str, default='.mp4', help='Input video format')
    parser.add_argument('--cores', type=int, default=24, help='Number of cores to use for multiprocessing')

    args = parser.parse_args()

    tasks = process_videos_in_folder(args.input_folder, args.output_folder, args.input_format)

    # Use multiprocessing Pool
    with Pool(args.cores) as pool:
        pool.starmap(process_video, tasks)

if __name__ == "__main__":
    main()
