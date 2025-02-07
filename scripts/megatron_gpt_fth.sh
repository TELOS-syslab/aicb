```sh
#!/bin/sh

set -x # 打开Shell的调试模式，显示执行的所有命令
: ${WORLD_SIZE:=1} # 默认定义分布式计算的节点数量，初始默认值为1
: ${RANK:=0} # 默认定义当前节点的排名，初始默认值为0
: ${MASTER_ADDR:="localhost"} # 默认定义进程主节点的地址，初始为本地主机地址
: ${MASTER_PORT:=29500} # 默认定义主节点通信的端口号，初始为29500
NUM_GPUS=$(nvidia-smi -L | wc -l) # 获取单节点的GPU数量
model_size=13 # 默认模型大小为13
num_layers=40 # 模型的层数
num_attention_heads=40 # 模型的注意力头数
hidden_size=5120 # 模型的隐藏层大小
seq_length=2048 # 输入序列的长度
micro_batch=1 # 微批次大小
epoch_num=1 # 训练的轮数
tensor_model_parallel_size=8 # 张量模型并行规模大小
pipeline_model_parallel =1 # 流水线模型并行规模
vocab_size=50257 # 词汇表大小
model_name=gpt_13b # 模型名称
ga_num=2 # 全局梯度累积数
sp_enable= # 是否启用序列并行
frame=Megatron # 使用的通信框架名称，默认Megatron
aiob_enable= # 是否启用异步输入输出包
max_position_embeddings=4096 # 最大位置嵌入维度
num_experts=1 # 使用的MoE（专家）数量
moe_enable= # 是否启用MoE（专家）
enable_visual= # 是否启用可视化
workload_only= # 仅生成工作负载
usage() {
  echo "Usage: \$0 [options] # 显示用法选项信息
    options:
      --frame              Communication framework: $frame # 定义使用的通信框架
      --world_size              World size (number of nodes): $WORLD_SIZE # 定义分布式计算的节点数量
      --tensor_model_parallel_size                  Tensor parallelism size: $tensor_model_parallel_size # 定义张量并行规模
      --pipeline_model_parallel                  Pipeline parallelism size: $pipeline_model_parallel # 定义流水线并行规模
      --global_batch            Global batch size: $global_batch # 定义全局批次大小
      --micro_batch             Micro batch size: $micro_batch # 定义微批次大小
      --num_layers              Number of layers: $num_layers # 定义模型层数
      --seq_length              Sequence length: $seq_length # 定义输入序列长度
      --hidden_size             Hidden size: $hidden_size # 定义隐藏层规模
      --epoch_num               Number of epochs: $epoch_num # 定义训练迭代数
      --num_attention_heads     Number of attention heads: $num_attention_heads # 定义注意力头数
      --aiob_enable             Enable AIOB: $aiob_enable # 是否启用异步输入输出包
      --enable_visual           Enable Visualization $enable_visual # 是否启用可视化
      --workload_only           generate workload only # 仅生成工作负载
      --use_flash_attn          Use flash attention: $use_flash_attn # 是否使用Flash注意力机制
      --swiglu                  Use SWIGLU: $swiglu # 是否使用SWIGLU
      --ffn_hidden_size         FFN hidden size: $ffn_hidden_size # 定义FFN隐藏层规模
      --comp_filepath           Computation file path: $comp_filepath # 计算文件路径
      --model_name              Model name: $model_name # 模型名称
      -m, --model_size          model size, defaults to $model_size (possible values: 175, 22, 13, 7) # 模型尺寸选项
      --max_position_embeddings Max position embeddings: $max_position_embeddings # 最大位置嵌入
      --nnodes                  Number of nodes: $WORLD_SIZE # 节点数量
      --node_rank               Rank of the node: $RANK # 节点的排名
      --nproc_per_node          Number of GPUs per node: $NUM_GPUS # 每个节点的GPU数量
      --master_addr             Master address: $MASTER_ADDR # 主节点地址
      --master_port             Master port: $MASTER_PORT # 主节点通信端口
      --me_enable                enable moe # 是否启用MoE
      --moe_router_topk         Number of experts to route to for each token. # token分配的专家数量
      --expert_model_parallel_size     Degree of expert model parallelism # 专家模型并行度
      --num_experts          Number of experts in the MoE model.  # MoE模型中的专家数量
      --moe_grouped_gemm        apply grouped gemm # 是否使用分组gemm
      -h, --help                Display this help and exit"1>&2; exit 1; # 显示帮助并退出
}

# 解析命令行参数，当参数数量大于0时执行循环
while [ $# -gt 0 ]
do
echo "Processing argument: $1" # 打印当前处理参数
  case $1 in # 匹配参数名
    --frame)
      frame=$2; shift;; # 设定通信框架，并跳过参数值
    --world_size)
      world_size=$2; shift;; # 设定节点数量，并跳过参数值
    --tensor_model_parallel_size|tp_num)
      tensor_model_parallel_size=$2; shift;; # 设定张量并行规模，并跳过参数值
    --pipeline_model_parallel|pp_num)
      pipeline_model_parallel=$2; shift;; # 设定流水线并行规模，并跳过参数值
    --global_batch)
      global_batch=$2; shift;; # 设定全局批次，并跳过参数值
    --micro_batch)
      micro_batch=$2; shift;; # 设定微批次，并跳过参数值
    --num_layers)
      num_layers=$2; shift;; # 设定模型层数，并跳过参数值
    --seq_length)
      seq_length=$2; shift;; # 设定序列长度，并跳过参数值
    --hidden_size)
      hidden_size=$2; shift;; # 设定隐藏层尺度，并跳过参数值
    --epoch_num)
      epoch_num=$2; shift;; # 设定训练迭代数，并跳过参数值
    --num_attention_heads)
      num_attention_heads=$2; shift;; # 设定注意力头数，并跳过参数值
    --aiob_enable)
      aiob_enable=--aiob_enable;; # 启用异步输入输出包
    --enable_visual)
      enable_visual=--enable_visual;; # 启用可视化选项
    --workload_only)
      workload_only=--workload_only;; # 启用仅生成工作负载模式
    --use_flash_attn)
      use_flash_attn=--use_flash_attn;; # 启用Flash Attention
    --swiglu)
      swiglu=--swiglu;; # 启用SWIGLU
    --ffn_hidden_size)
      ffn_hidden_size=$2; shift;; # 设定FFN隐藏层尺度，并跳过参数值
    --sp|--sp-enable|--enable_sequence_parallel)
      sp_enable=--enable_sequence_parallel;; # 启用序列并行
    --comp_filepath)
      comp_filepath=$2; shift;; # 指定计算文件路径，并跳过参数值
    -m|--model_size)
      model_size=$2; shift;; # 设定模型尺寸，并跳过参数值
    --moe_enable)
      moe_enable=--moe_enable;; # 启用MoE
    --moe_router_topk|--topk)
      moe_router_topk=$2; shift;; # 设定MoE路由的topk值，并跳过参数值
    --num_experts|--experts)
      num_experts=$2; shift;; # 设定MoE专家数量，并跳过参数值
    --expert_model_parallel_size|--ep)
      expert_model_parallel_size=$2; shift;; # 设定专家模型并行度，并跳过参数值
    --grouped_gemm|--moe_grouped_gemm)
      grouped_gemm=--moe_grouped_gemm;; # 启用分组GEMM功能
    --nnodes)
      WORLD_SIZE=$2;shift;; # 设定节点数量，并跳过参数值
    --node_rank)
      RANK=$2;shift;; # 设定节点排名，并跳过参数值
    --nproc_per_node)
      NUM_GPUS=$2;shift;; # 设定每节点的GPU数量，并跳过参数值
    --master_addr)
      MASTER_ADDR=$2;shift;; # 设定Master地址，并跳过参数值
    --master_port)
      MASTER_PORT=$2;shift;;  # 设定Master端口，并跳过参数值
    -h|--help)
      usage ;; # 打印帮助信息并退出
    (*)
      break;; # 处理未定义参数名
  esac

  shift # 跳出循环或继续处理下一个参数
done

# 根据模型尺寸设置相关参数
case $model_size in
  175)
    model_name=gpt_175B
    num_layers=96
    hidden_size=12288
    num_attention_heads=96
    tensor_model_parallel_size=8
    ;;
  22)
    model_name=gpt_22B
    num_layers=48
    hidden_size=6144
    num_attention_heads=64
    tensor_model_parallel_size=8
    ;;
  13)
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
  7)
    model_name=gpt_7B
    num_layers=36
    hidden_size=4096
    num_attention_heads=32
    ;;
  405)
    model_name=llama_405B
    num_layers=128
    hidden_size=16384
    ffn_hidden_size=53248
    num_attention_heads=128
    tensor_model_parallel_size=8
    pipeline_model_parallel=16
    ;;
  65)
    model_name=llama_65B
    num_layers=80
    hidden_size=8192
    ffn_hidden_size=28672
    num_attention_heads=64
    tensor_model_parallel_size=8
    pipeline_model_parallel=2
    ;;
  moe)
    model_name=Mixtral_8*7B
    num_layers=32
    hidden_size=4096
    num_attention_heads=32
    ffn_hidden_size=14336
    tensor_model_parallel_size=2
    moe_enable=--moe_enable
    grouped_gemm=--moe_grouped_gemm
    ;;
  (*)
    echo "Only support model size 405,175,22,13,7 or moe; using default size 13"
    model_name=gpt_13B
    num_layers=40
    hidden_size=5120
    num_attention_heads=40
    ;;
esac

dp_num=$((world_size/tensor_model_parallel_size/pipeline_model_parallel)) # 计算数据并行数
global_batch=$((ga_num*dp_num*micro_batch)) # 计算全局批次大小
if [ $workload_only ]; then
  script="python -m workload_generator.generate_megatron_workload" # 设置为生成工作负载脚本
else
  script="./aicb.py" # 默认运行aicb.py脚本
fi

cmd="$script \ # 构建执行命令
  --frame=$frame \ # 框架参数
  --model_name=$model_name \ # 模型名称
  --world_size=$(($WORLD_SIZE * $NUM_GPUS)) \ # 计算总世界大小并设置
  --tensor_model_parallel_size=$tensor_model_parallel_size \ # 张量并行参数
  --micro_batch=$micro_batch \ # 微批次设置
  --global_batch=$global_batch \ # 全局批次设置
  --epoch_num=$epoch_num \ # 训练轮数
  --num_layers=$num_layers \ # 模型层数
  --hidden_size=$hidden_size \ # 隐藏层尺寸
  --num_attention_heads=$num_attention_heads \ # 注意力头数
  --seq_length=$seq_length \ # 序列长度
  --vocab_size=$vocab_size \ # 词汇表大小
  --pipeline_model_parallel=$pipeline_model_parallel \ # 流水线并行规模
  --use-distributed-optimizer \ # 启用分布式优化
  --max_position_embeddings=$max_position_embeddings \ # 最长位置嵌入设置
  ${aiob_enable} \ # 启用/禁用异步输入输出包
  ${enable_visual} \ # 启用/禁用可视化
  ${workload_only} \ # 工作负载模式选项
  ${sp_enable} \ # 启用/禁用序列并行
  ${use_flash_attn} \ # 启用/禁用Flash Attention
  ${swiglu} \ # 启用/禁用SWIGLU
  ${ffn_hidden_size:+--ffn_hidden_size=$ffn_hidden_size} \ # 设置FFN隐藏层大小
  ${comp_filepath:+--comp_filepath=$comp_filepath} \ # 设置计算文件路径
  ${moe_enable} \ # 启用/禁用MoE
  ${moe_router_topk:+--moe_router_topk=$moe_router_topk} \ # 设置MoE路由topk参数
  ${num_experts:+--num_experts=$num_experts} \ # 设置MoE专家数量
  ${expert_model_parallel_size:+--expert_model_parallel_size=$expert_model_parallel_size} \ # 设置专家模型并行度
  ${grouped_gemm}" # 启用/禁用分组GEMM
echo $cmd # 输出命令到控制台

if [ $workload_only ]; then
  $cmd # 仅生成工作负载模式下直接运行命令
else
  torchrun \ # 使用torchrun启动训练
    --nnodes $WORLD_SIZE \ # 节点数量参数
    --node_rank $RANK \ # 节点排名参数
    --nproc_per_node $NUM_GPUS \ # 每个节点GPU数量
    --master_addr $MASTER_ADDR \ # 主节点地址
    --master_port $MASTER_PORT \ # 主节点端口
    $cmd # 执行命令
fi # 结束if语句
```