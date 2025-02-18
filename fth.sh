======250207

sudo docker run --gpus=all -d -it --privileged --user=root --net=host --ipc=host --name=fth_simai --device=/dev/infiniband/ --ulimit memlock=-1:-1 --shm-size=180g -v /disk1/futianhao:/app nvcr.io/nvidia/pytorch:24.09-py3
sudo docker exec -it fth_simai /bin/bash


cd /app/software1/aicb/

watch -n 1 nvidia-smi

nvidia-smi -l 1

======250208

dpkg -i /download/AICB_v1.0.deb 

sh scripts/megatron_workload_with_aiob.sh -m 7

==========250218====

docker exec -it fth_simai03 bash

cd /app/software1/aicb/

python -m workload_generator.AIOB_simAI_workload_generator --gpu_type=H20 --frame=Megatron --world_size=9216 --tensor_model_parallel_size=2 --pipeline_model_parallel=12  --expert_model_parallel_size=64  --global_batch=9216 --micro_batch=1 --num_layers=96 --seq_length=4096 --hidden_size=6144 --epoch_num=10 --num_attention_heads=64 --model_name=MoE_large --max_position_embeddings=4096 --vocab_size=151936 --use-distributed-optimizer --aiob_enable --use_flash_attn --swiglu --enable_sequence_parallel --recompute_activations --ffn_hidden_size=7168 --num_experts=192 --moe_grouped_gemm --moe_enable --moe_router_topk=4

python -m workload_generator.AIOB_vidur_simAI_workload_generator

不会真正运行megatron； 把megatron抄过来了；
不会真正运行sarathi； 把sarathi抄过来了；