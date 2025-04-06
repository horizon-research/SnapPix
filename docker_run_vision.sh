#!/bin/bash
set -e

docker run --runtime nvidia \
    --name coded_test \
    -d -it \
    --network=host \
    --volume /mnt/Yu_SSD:/workspace \
    dustynv/sam:r35.3.1
docker start  coded_test
echo "Container nerf_vr_test is running in the background."
echo "To attach to the container, use 'sudo docker attach nerf_vr_test'"
echo "To execute a command inside the container, use 'sudo docker exec -it nerf_vr_test <command>'"