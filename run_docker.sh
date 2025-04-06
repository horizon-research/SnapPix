# docker run --gpus all \
#            --name coded_exposure \
#            -d -it \
#            --network=host \
#            --privileged \
#            --volume /localdisk/wk_coded_exposure/:/workspace \
#            --volume /localdisk2/:/localdisk2/ \
#            --shm-size=128g \
#            pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
docker run --gpus all \
           --name coded_exposure \
           -d -it \
           --network=host \
           --privileged \
           --volume /home/lwk/ur_research/CodedExposure:/workspace \
           --shm-size=128g \
           pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel