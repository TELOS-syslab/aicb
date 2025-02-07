======250207

sudo docker run --gpus=all -d -it --privileged --user=root --net=host --ipc=host --name=fth_simai --device=/dev/infiniband/ --ulimit memlock=-1:-1 --shm-size=180g -v /disk1/futianhao:/app nvcr.io/nvidia/pytorch:24.09-py3
sudo docker exec -it fth_simai /bin/bash


cd /app/software1/aicb/

watch -n 1 nvidia-smi

nvidia-smi -l 1

======250208

dpkg -i /download/AICB_v1.0.deb 

sh scripts/megatron_workload_with_aiob.sh -m 7