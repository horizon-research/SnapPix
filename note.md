- 
    - add VideoMAEv2
    - config environment
        - cuda 12.2
        - requirements.txt
    - Construct dataset
        - K710 + SSV2
        - K710: see https://github.com/cvdfoundation/kinetics-datWaset
        - SSV2: use my previous prepared
- 
    - Corrupted Data replacement: python3 kinetic_replacement.py kinetics-dataset/k400/replacement/replacement_for_corrupted_k400/ kinetics-dataset/k400/train/ kinetics-dataset/k400/val/ kinetics-dataset/k400/test/
    - python3 kinetic_downsampler.py kinetics-dataset/k400 kinetics-dataset/k600 kinetics-dataset/k700 kinetics-dataset/downsampled/k400 kinetics-dataset/downsampled/k600 kinetics-dataset/downsampled/k700

- tar -cvf - /localdisk/wk_coded_exposure/CodedExposure/dataset/k710/train | split -b 20G·················· -d - compressed_train.tar.gz.part

SSV2
- git clone https://huggingface.co/datasets/WK1997/ssv2
- unzip '*.zip'
- cat 20bn-something-something-v2-?? | tar zx


- Just use open mmlab for all dataset: https://opendatalab.com/OpenMMLab
- Data Preprocessing (Downsample + Inverse Gamma + Grayscale): python preprocess_data.py ../../OpenDataLab___sthv2/raw/sthv2/sthv2/videos/ ssv2_processed/ --input_format .webm
- python3 combine_pretrained.py dataset_lists/K710/train.csv dataset_lists/SSV2/train.csv mmdataset/k400_processed mmdataset/k600_processed mmdataset/k700_processed mmdataset/ssv2_processed combined_pretrain



