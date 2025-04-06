sudo systemctl stop gdm

# 4090
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -i 0 --lock-memory-clocks=10501,10501
sudo nvidia-smi -i 0 --lock-gpu-clocks=2805,2805 # (3165 is the maximum, but can’t be set in real case)
sudo nvidia-smi -pl 530

# jetson
sudo jetson_clocks
