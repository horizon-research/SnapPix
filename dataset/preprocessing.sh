python preprocess_data.py mmdataset/OpenDataLab___sthv2/raw/sthv2/sthv2/videos/ mmdataset/ssv2_processed/ --input_format .webm
python preprocess_data.py mmdataset/OpenDataLab___UCF101/raw/UCF101/source_compressed/data/UCF-101 mmdataset/ucf_processed/ --input_format .avi
python preprocess_data.py mmdataset/OpenMMLab___Kinetics-400/raw/Kinetics-400 mmdataset/k400_processed/ --input_format .mp4
python preprocess_data.py mmdataset/OpenMMLab___Kinetics600/raw/Kinetics600/videos mmdataset/k600_processed/ --input_format .mp4
python preprocess_data.py mmdataset/OpenMMLab___Kinetics_700-2020/raw/Kinetics_700-2020/kinetics-dataset/k700-2020_targz/train mmdataset/k700-2020_processed/ --input_format .mp4
python preprocess_data.py mmdataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/videos mmdataset/k700_processed/ --input_format .mp4