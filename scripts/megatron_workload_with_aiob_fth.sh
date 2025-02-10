#!/bin/sh  # 指定脚本使用的解释器为 sh

# 定义默认的配置参数
frame=Megatron  # 通信框架，默认为 Megatron
world_size=32  # 分布式训练的世界规模（总进程数），默认为 32
tensor_model_parallel_size=8  # 张量模型并行规模，默认为 8
pipeline_model_parallel=1  # 管道模型并行规模，默认为 1
global_batch=1024  # 全局批次大小，默认为 1024
micro_batch=1  # 微批次大小，默认为 1
num_layers=40  # 模型层数，默认为 40
seq_length=4096  # 序列长度，默认为 4096
hidden_size=5120  # 隐藏层大小，默认为 5120
epoch_num=1  # 训练轮数，默认为 1
num_attention_heads=40  # 注意力头数量，默认为 40
aiob_enable=  # 是否启用 AIOB，默认未启用
use_flash_attn=  # 是否使用 Flash Attention，默认未启用
swiglu=  # 是否使用 SwiGLU 激活函数，默认未启用
sp_enable=  # 是否启用序列并行，默认未启用
ffn_hidden_size=  # 前馈网络隐藏层大小，默认未设置
comp_filepath=  # 计算文件路径，默认未设置
model_size=13  # 模型大小，默认为 13（代表 gpt_13B）
max_position_embeddings=4096  # 最大位置嵌入数，默认为 4096
vocab_size=50257  # 词汇表大小，默认为 50257
num_experts=1  # MoE 模型中的专家数量，默认为 1
moe_enable=  # 是否启用 Mixture of Experts，默认未启用
recompute_activations=  # 是否重新计算激活函数，默认未启用
gpu_type=None  # GPU 类型，默认为 None

# 定义使用说明函数
usage() {
  echo "Usage: \$0 [options]
    options:
      --frame              communication framework, defaults to $frame  # 通信框架，默认值
      --world_size              world size, defaults to $world_size  # 世界规模，默认值
      --tensor_model_parallel_size                  tensor parallelism size, defaults to $tensor_model_parallel_size  # 张量并行规模，默认值
      --pipeline_model_parallel                  pipeline parallelism size, defaults to $pipeline_model_parallel  # 管道并行规模，默认值
      --global_batch            global batch size, defaults to $global_batch  # 全局批次大小，默认值
      --micro_batch             micro batch size, defaults to $micro_batch  # 微批次大小，默认值
      --num_layers              number of layers, defaults to $num_layers  # 层数，默认值
      --seq_length              sequence length, defaults to $seq_length  # 序列长度，默认值
      --hidden_size             hidden size, defaults to $hidden_size  # 隐藏层大小，默认值
      --epoch_num               number of epochs, defaults to $epoch_num  # 训练轮数，默认值
      --use_distributed_optimizer use distributed optimizer  # 是否使用分布式优化器
      --num_attention_heads     number of attention heads, defaults to $num_attention_heads  # 注意力头数量，默认值
      --aiob_enable             enable AIOB  # 是否启用 AIOB
      --use_flash_attn          use flash attention  # 是否使用 Flash Attention
      --swiglu                  use swiglu  # 是否使用 SwiGLU 激活函数
      --ffn_hidden_size         FFN hidden size  # 前馈网络隐藏层大小
      --comp_filepath           computation file path  # 计算文件路径
      --max_position_embeddings max position embeddings, defaults to $max_position_embeddings  # 最大位置嵌入数，默认值
      -m, --model_size          model size, defaults to $model_size (possible values: 175, 22, 13, 7, moe)  # 模型大小，默认值及可选值
      --moe_enable             enable moe  # 是否启用 Mixture of Experts
      --moe_router_topk         Number of experts to route to for each token.  # 每个 token 路由到的专家数量
      --expert_model_parallel_size     Degree of expert model parallelism  # 专家模型并行规模
      --num_experts          Number of experts in the MoE model.  # MoE 模型中的专家数量
      --moe_grouped_gemm        apply grouped gemm  # 是否应用分组 GEMM
      -h, --help                display this help and exit" 1>&2; exit 1;  # 显示帮助信息并退出
}

# 解析命令行参数
while [ $# -gt 0 ]
do
  case $1 in
    --gpu_type)  # 处理 --gpu_type 参数
      gpu_type=$2; shift;;  # 设置 gpu_type 变量，并移位
    --frame)  # 处理 --frame 参数
      frame=$2; shift;;  # 设置 frame 变量，并移位
    --world_size)  # 处理 --world_size 参数
      world_size=$2; shift;;  # 设置 world_size 变量，并移位
    --tensor_model_parallel_size|--tp)  # 处理 --tensor_model_parallel_size 或 --tp 参数
      tensor_model_parallel_size=$2; shift;;  # 设置 tensor_model_parallel_size 变量，并移位
    --pipeline_model_parallel|--pp)  # 处理 --pipeline_model_parallel 或 --pp 参数
      pipeline_model_parallel=$2; shift;;  # 设置 pipeline_model_parallel 变量，并移位
    --global_batch)  # 处理 --global_batch 参数
      global_batch=$2; shift;;  # 设置 global_batch 变量，并移位
    --micro_batch)  # 处理 --micro_batch 参数
      micro_batch=$2; shift;;  # 设置 micro_batch 变量，并移位
    --num_layers)  # 处理 --num_layers 参数
      num_layers=$2; shift;;  # 设置 num_layers 变量，并移位
    --seq_length)  # 处理 --seq_length 参数
      seq_length=$2; shift;;  # 设置 seq_length 变量，并移位
    --hidden_size)  # 处理 --hidden_size 参数
      hidden_size=$2; shift;;  # 设置 hidden_size 变量，并移位
    --epoch_num)  # 处理 --epoch_num 参数
      epoch_num=$2; shift;;  # 设置 epoch_num 变量，并移位
    --num_attention_heads)  # 处理 --num_attention_heads 参数
      num_attention_heads=$2; shift;;  # 设置 num_attention_heads 变量，并移位
    --aiob_enable|--aiob)  # 处理 --aiob_enable 或 --aiob 参数
      aiob_enable=--aiob_enable;;  # 设置 aiob_enable 变量
    --use_flash_attn|--flash_attn)  # 处理 --use_flash_attn 或 --flash_attn 参数
      use_flash_attn=--use_flash_attn;;  # 设置 use_flash_attn 变量
    --swiglu)  # 处理 --swiglu 参数
      swiglu=--swiglu;;  # 设置 swiglu 变量
    --ffn_hidden_size)  # 处理 --ffn_hidden_size 参数
      ffn_hidden_size=$2; shift;;  # 设置 ffn_hidden_size 变量，并移位
    --sp|--sp-enable)  # 处理 --sp 或 --sp-enable 参数
      sp_enable=--enable_sequence_parallel;;  # 设置 sp_enable 变量
    --comp_filepath)  # 处理 --comp_filepath 参数
      comp_filepath=$2; shift;;  # 设置 comp_filepath 变量，并移位
    -m|--model_size)  # 处理 -m 或 --model_size 参数
      model_size=$2; shift;;  # 设置 model_size 变量，并移位
    --max_position_embeddings)  # 处理 --max_position_embeddings 参数
      max_position_embeddings=$2; shift;;  # 设置 max_position_embeddings 变量，并移位
    --moe_enable)  # 处理 --moe_enable 参数
      moe_enable=--moe_enable;;  # 设置 moe_enable 变量
    --moe_router_topk|--topk)  # 处理 --moe_router_topk 或 --topk 参数
      moe_router_topk=$2; shift;;  # 设置 moe_router_topk 变量，并移位
    --num_experts|--experts)  # 处理 --num_experts 或 --experts 参数
      num_experts=$2; shift;;  # 设置 num_experts 变量，并移位
    --expert_model_parallel_size|--ep)  # 处理 --expert_model_parallel_size 或 --ep 参数
      expert_model_parallel_size=$2; shift;;  # 设置 expert_model_parallel_size 变量，并移位
    --grouped_gemm|--moe_grouped_gemm)  # 处理 --grouped_gemm 或 --moe_grouped_gemm 参数
      grouped_gemm=--moe_grouped_gemm;;  # 设置 grouped_gemm 变量
    --recompute_activations|--recompute)  # 处理 --recompute_activations 或 --recompute 参数
      recompute_activations=--recompute_activations;;  # 设置 recompute_activations 变量
    -h|--help)  # 处理 -h 或 --help 参数
      usage;;  # 调用使用说明函数
    (*)  # 处理其他未识别的参数
      break;;  # 退出参数解析
  esac
  shift  # 移动到下一个参数
done

# 根据模型大小设置相关参数
case $model_size in
  175)  # 如果模型大小为175
    model_name=gpt_175B  # 设置模型名称为 gpt_175B
    num_layers=96  # 设置层数为96
    hidden_size=12288  # 设置隐藏层大小为12288
    num_attention_heads=96  # 设置注意力头数量为96
    tensor_model_parallel_size=8  # 设置张量并行规模为8
    ;;
  22)  # 如果模型大小为22
    model_name=gpt_22B  # 设置模型名称为 gpt_22B
    num_layers=48  # 设置层数为48
    hidden_size=6144  # 设置隐藏层大小为6144
    num_attention_heads=64  # 设置注意力头数量为64
    tensor_model_parallel_size=8  # 设置张量并行规模为8
    ;;
  13)  # 如果模型大小为13
    model_name=gpt_13B  # 设置模型名称为 gpt_13B
    num_layers=40  # 设置层数为40
    hidden_size=5120  # 设置隐藏层大小为5120
    num_attention_heads=40  # 设置注意力头数量为40
    ;;
  7)  # 如果模型大小为7
    model_name=gpt_7B  # 设置模型名称为 gpt_7B
    num_layers=36  # 设置层数为36
    hidden_size=4096  # 设置隐藏层大小为4096
    num_attention_heads=32  # 设置注意力头数量为32
    tensor_model_parallel_size=4  # 设置张量并行规模为4
    ;;
  405)  # 如果模型大小为405
    model_name=llama_405B  # 设置模型名称为 llama_405B
    num_layers=128  # 设置层数为128
    hidden_size=16384  # 设置隐藏层大小为16384
    ffn_hidden_size=53248  # 设置前馈网络隐藏层大小为53248
    num_attention_heads=128  # 设置注意力头数量为128
    ;;
  moe)  # 如果模型大小为moe
    model_name=Mixtral_8*7B  # 设置模型名称为 Mixtral_8*7B
    num_layers=32  # 设置层数为32
    hidden_size=4096  # 设置隐藏层大小为4096
    num_attention_heads=32  # 设置注意力头数量为32
    ffn_hidden_size=14336  # 设置前馈网络隐藏层大小为14336
    tensor_model_parallel_size=4  # 设置张量并行规模为4
    moe_enable=--moe_enable  # 启用 MoE
    grouped_gemm=--moe_grouped_gemm  # 启用分组 GEMM
    ;;
  (*)  # 如果模型大小不在上述选项中
    echo "Only support model size 175, 22,13 or 7; using default size 13"  # 输出支持的模型大小并告知使用默认大小13
    model_name=gpt_13B  # 设置模型名称为 gpt_13B
    num_layers=40  # 设置层数为40
    hidden_size=5120  # 设置隐藏层大小为5120
    num_attention_heads=40  # 设置注意力头数量为40
    ;;
esac

# 构建运行命令
cmd="python -m workload_generator.AIOB_simAI_workload_generator \  # 调用 Python 模块 workload_generator.AIOB_simAI_workload_generator
  --gpu_type=$gpu_type \  # 传递 gpu_type 参数
  --frame=$frame \  # 传递 frame 参数
  --world_size=$world_size \  # 传递 world_size 参数
  --tensor_model_parallel_size=$tensor_model_parallel_size \  # 传递 tensor_model_parallel_size 参数
  --pipeline_model_parallel=$pipeline_model_parallel \  # 传递 pipeline_model_parallel 参数
  --global_batch=$global_batch \  # 传递 global_batch 参数
  --micro_batch=$micro_batch \  # 传递 micro_batch 参数
  --num_layers=$num_layers \  # 传递 num_layers 参数
  --seq_length=$seq_length \  # 传递 seq_length 参数
  --hidden_size=$hidden_size \  # 传递 hidden_size 参数
  --epoch_num=$epoch_num \  # 传递 epoch_num 参数
  --num_attention_heads=$num_attention_heads \  # 传递 num_attention_heads 参数
  --model_name=$model_name \  # 传递 model_name 参数
  --max_position_embeddings=$max_position_embeddings \  # 传递 max_position_embeddings 参数
  --vocab_size=$vocab_size \  # 传递 vocab_size 参数
  --use-distributed-optimizer \  # 启用分布式优化器
  ${aiob_enable} \  # 传递 aiob_enable 参数（如果启用）
  ${use_flash_attn} \  # 传递 use_flash_attn 参数（如果启用）
  ${swiglu} \  # 传递 swiglu 参数（如果启用）
  ${sp_enable} \  # 传递 sp_enable 参数（如果启用）
  ${recompute_activations} \  # 传递 recompute_activations 参数（如果启用）
  ${ffn_hidden_size:+--ffn_hidden_size=$ffn_hidden_size} \  # 如果设置了 ffn_hidden_size，则传递相应参数
  ${comp_filepath:+--comp_filepath=$comp_filepath} \  # 如果设置了 comp_filepath，则传递相应参数
  ${moe_enable} \  # 传递 moe_enable 参数（如果启用）
  ${moe_router_topk:+--moe_router_topk=$moe_router_topk} \  # 如果设置了 moe_router_topk，则传递相应参数
  ${num_experts:+--num_experts=$num_experts} \  # 如果设置了 num_experts，则传递相应参数
  ${expert_model_parallel_size:+--expert_model_parallel_size=$expert_model_parallel_size} \  # 如果设置了 expert_model_parallel_size，则传递相应参数
  ${grouped_gemm} " \  # 传递 grouped_gemm 参数（如果启用）
  
echo $cmd  # 输出构建的命令

$cmd  # 执行构建的命令
