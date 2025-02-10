# 简介
AI通信基准测试（AI Communication Benchmark）是一个专为人工智能（AI）场景设计的通信基准测试套件，主要用于评估通信栈的性能。该套件不仅提供详细的性能指标，还帮助开发者快速识别和诊断通信栈中潜在的性能瓶颈和问题。通过模拟AI训练和推理过程中的实际通信流量模式，该基准测试能够准确反映在高并发和大数据传输量条件下的通信栈实际性能。无论是分布式计算中的节点间通信，还是大规模模型训练中的数据同步，该基准测试套件都提供了有效的性能评估和优化建议，以帮助用户提升整体系统的效率和稳定性。

# 环境搭建
在搭建环境之前，首先将代码仓库拉取到本地，然后进行环境配置：
```
git clone https://github.com/aliyun/aicb.git
```
对于环境，如果您只是生成工作负载，则不需要额外的依赖项。然而，其他功能需要依赖如PyTorch、CUDA、NCCL和NVIDIA APEX等库。因此，您可以通过配置本地环境或使用Docker来设置适当的运行环境。
## 使用Dockerfile设置环境
```
docker build -t aicb:v0.0.1 .
docker run --gpus all --net host --shm-size 16g -it --rm aicb:v0.0.1 
```
## 本地环境搭建
对于本地环境，您需要 Python >= 3.8、CUDA 版本 >= 11.8、PyTorch >= 2.0.0 和 NVIDIA APEX。
## 使用官方Docker
您还可以使用 [NGC的PyTorch容器](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) 创建所需的Docker环境，其中 pytorch:xx.xx 应该 >= pytorch:23.08。
```
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/AICBench:/workspace/AICBench nvcr.io/nvidia/pytorch:xx.xx-py3
```

# 基本用法
## 物理执行
在物理机上运行时，需要对PyTorch相关的环境变量进行额外配置。这可以通过在脚本中显式指定它们或将环境变量直接添加到系统中完成。下表列出了所需的环境变量：

| 参数名称       | 描述                   |
|----------------|------------------------|
| nnodes         | 节点数量               |
| node_rank      | 节点的排名编号         |
| nproc_per_node | 每个节点的GPU数量      |
| master_addr    | 主节点地址             |

### 单节点执行快速入门
在物理机上运行AICB的脚本是：[/scripts/megatron_workload_with_aiob.sh](../scripts/megatron_workload_with_aiob.sh)

我们提供了四种预置模型（7/13/22/175B）和MoE模型，可以快速启动并运行在物理机上，可以通过参数`--model_size`指定。此外，Megatron并行框架支持启用`aiob_enable`选项，以获取实际模型每个操作的计算时间。如果不使用aiob，则只能填充固定等待时间。或者，当启用AIOB时，您可以指定`--comp_filepath`以填充相应的计算时间。
以下是一个生成模型大小为13B、tp 8、pp 1、总GPU数为8、gbs 2、mbs 1、序列长度为4096，并启用flash_attn和swiglu的Workload示例，同时使用AIOB获取实际模型每个操作的计算时间。
```bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=23089
export WORLD_SIZE=1
export RANK=0

sh ./scripts/megatron_gpt.sh \
-m 13 --world_size 8 --tensor_model_parallel_size 8 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 2  \
--micro_batch 1 --seq_length 4096 \
--swiglu --use_flash_attn  --aiob_enable  
```

### 多节点执行快速入门
用于多节点执行的脚本是：[run_in_cluster.py](../scripts/run_in_cluster.py)

步骤：
1. 首先安装批量分发命令（如pssh和pscp）。
2. 编辑要使用的集群的`iplist`文件，每行添加一台机器的可访问IP地址。
3. 修改[run_in_cluster.py](../scripts/run_in_cluster.py)，指定镜像名称以及`iplist`文件和AICB主目录的路径。更多细节请参考[run_in_cluster.py](../scripts/run_in_cluster.py)的文档。
4. 修改[run_suites.py](../run_suites.py)，选择要运行的工作负载（默认：无工作负载）。
5. 将`iplist`和AICB源代码复制到每台机器（例如，使用pscp）。
4. 运行命令如下：`pssh -i -h /path/to/iplist -o out -e err -t 0 "cd /path/to/aicb && python run_in_cluster.py"`。记得将`/path/to/iplist`和`/path/to/aicb`替换为您机器上的实际路径。

每台机器上运行的具体命令可以在突出显示的部分修改：

![扩展图](../images/tutorial_7.png)

### 日志与结果
每次通信完成后，程序将打印此次通信的相关日志。输出格式如下，几乎包含所有通信操作的信息：
* 通信类型
* 通信组
* 消息大小
* 通信执行时间
* [吞吐量](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

![扩展图](../images/tutorial_1.png)

所有通信完成后，将汇总关于用例的信息：
* 包括总体运行时间和每次迭代的时间分析，可以清楚地查看每次迭代是否正常运行以及是否存在抖动。
![扩展图](../images/tutorial_2.png)
* 每种通信类型的时间
首先区分模型训练阶段：初始化阶段和训练阶段。然后总结每个阶段中执行的集合通信，包括其对应的消息大小、频率以及具体的平均延迟、最大值和最小值等。这有助于确定哪个消息段的集合通信操作导致了异常，便于进一步调查和故障排除。
![扩展图](../images/tutorial_3.png)

#### 文件输出
文件输出包括两种不同类型的文件：.csv文件。
1. CSV文件保存在：
`results/comm_logs/megatron_gpt_13B_8n_log.csv`，您还可以看到执行时间、执行阶段以及属于不同`comm_group`和`comm_type`的算法带宽（algbw）和总线带宽（busbw）。它还包括模型每个部分的计算时间和所属的计算阶段。

![扩展图](../images/tutorial_4.png)

除了上述详细信息外，还提供了一个.csv文件用于详细分析结果。以下是使用方法：
    1. 读取_workload.csv日志：
      * 可以通过调用`log_analyzer.log.Workload.load(filename)`读取_workload.csv日志文件。
      * 这将返回Workload和args。
        * args包含用于训练输入的参数。
        * Workload由生成的中间结果组成。
    2. 读取_log.csv日志：
    * 可以通过调用`log_analyzer.log.Log.load(filename)`读取_log.csv日志文件。
    * 这将返回一个Log对象，包含：
      * comm_logs: List[LogItem]: 这是所有生成日志的列表。
      * epoch_times: List[int]: 这是每次迭代所花费的时间列表。第一次迭代通常表示初始化，可能会显示与后续迭代不同的通信行为，可能导致时间差异。
      * comm_log_each_epoch: List[List[LogItem]]: 这是一个列表，其中每个项目对应每次迭代的通信日志。如果某次迭代的时间与其他迭代显著不同，您可以分析此次迭代以识别导致差异的通信。
通过利用这些日志文件和解析方法，您可以对训练过程进行全面而详细的分析，识别任何异常或优化领域。

## 为仿真生成工作负载（SimAI）
### 快速入门
AICB生成工作负载的脚本是：`./scripts/megatron_workload_with_aiob.sh`
我们提供了四种预置模型（7/13/22/175B），可以快速生成相应的工作负载，可以通过参数`--model_size`指定。模型的计算部分可以选择通过`--aiob_enable`使用aiob。当不使用aiob时，默认使用固定时间填充工作负载。当启用`--aiob_enable`时，如果不指定`--comp_filepath`，则使用当前GPU的计算时间填充工作负载。
以下是一个生成模型大小为7B、tp 4、pp 1、总GPU数为4096、gbs 8192、mbs 1、序列长度为4096，并启用flash_attn、swiglu和aiob，同时读取Example.txt作为计算时间的工作负载示例。
```bash
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 4096 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt
```

### 工作负载
生成的工作负载结果保存在：`results/workload/gpt_7B-world_size4096-tp4-pp1-gbs8192-mbs1-seq4096-flash_attn-True.txt`
![扩展图](../images/tutorial_6.png)

## 使用自定义案例运行AICB
除了快速入门选项外，您还可以详细自定义模型参数，以在物理机上运行或生成所需的仿真和分析工作负载。这种灵活性允许您根据需求定制工作负载，无论您是在尝试不同配置的大语言模型、测试各种并行框架，还是优化运行时环境。自定义参数提供了对基准测试和仿真过程的更深入洞察和更大控制力，从而实现更精确的性能调优和分析。

### 参数
AICB的主要参数如下：

| 类别                     | 参数名称                   | 描述                                                                 | 
|------------------------------|-----------------------------------|-----------------------------------------------------------------------------|
| 名称                         | frame                             | DeepSpeed/Megatron                                                          |
|                              | model_name                        | Llama/GPT/...                                                               |
| 训练参数           | world_size                        | GPU总数                                                        |
|                              | global_batch                      | 训练的总批处理大小                                               |
|                              | micro_batch                       | 每个模型实例的批处理大小（本地批处理大小）。                           |
|                              | epoch_num                         | 迭代次数                                                        |
| 模型参数             | model_size                        | 模型大小（7/13/65/175/270B）和moe                                       |
|                              | num_layers                        | Transformer层数。                                               |
|                              | hidden_size                       | Transformer隐藏层大小。                                                    |
|                              | num_attention_heads               | Transformer注意力头数。                                      |
|                              | seq_length                        | 最大序列长度。                                         |
|                              | vocab_size                        | 在EOD或填充之前的词汇表大小。                                        |
|                              | max_position_embeddings           | 使用的最大位置嵌入数。                               |
|                              | ffn_hidden_size                   | Transformer前馈网络隐藏层大小。                               |
| Megatron并行参数 | tensor_model_parallel_size        | 张量模型并行度。                                         |
|                              | pipeline_model_parallel           | 流水线模型并行度。                                       |
|                              | enable_sequence_parallel          | 启用序列并行优化。                                      |
| Megatron优化参数 | use_flash_attn                | 使用FlashAttention实现注意力机制。                             |
|                              | swiglu                            | 使用门控线性单元和SiLU激活代替默认的gelu          |
|                              | openai_gelu                       | 使用OpenAI的GeLU实现。                                           |
|                              | onnx_safe                         | 使用解决已知Torch ONNX导出器问题的方法                 |
|                              | squared_relu                      | 使用平方ReLU激活代替默认的gelu                         |
|                              | bias_gelu_fusion                  | 启用偏置和gelu融合。                                                |
|                              | gated_linear_unit                 | 当使用swiglu时启用                                                      |
| MoE                          | expert_model_parallel_size        | 专家模型并行度                                          |
|                              | moe_enable                        | 启用MoE                                                                  |
|                              | num_experts                       | MoE中的专家数量（None表示没有MoE）                                |
|                              | moe_router_topk                   | 每个token路由到的专家数量。                               |
|                              | moe_grouped_gemm                  | 当每个rank有多个专家时，压缩多个本地（可能较小的）gemm到单个内核中|
| DeepSpeed参数         | zero_stage,reduce_bucket_size     | 选择zero优化器阶段                                                 |
|                              | allgather_bucket_size             | 优化all-gather操作期间的通信效率和内存使用，仅适用于第1/2阶段                                                          |
|                              | prefetch_bucket_size, param_persistence_threshold, model_persistence_threshold, max_live_parameters | 仅适用于第3阶段。控制预取参数的数量。控制all_gather和reduce_scatter的大小 |
| 其他                        | aiob_enable                       | 启用AIOB以获取计算时间                                      |
|                              | comp_filepath                     | 使用aiob_lib获取操作计算时间                                  |

### 在物理GPU集群上运行
当前运行自定义案例的入口文件是[aicb.py](../aicb.py)。通过使用此文件，您可以灵活选择更多参数进行调优。
```bash
# DeepSpeed Stage 2 示例
torchrun \
--nnodes ${WORLD_SIZE} \
--node_rank ${RANK} \
--nproc_per_node gpu \
--master_addr ${MASTER_ADDR} \
--master_port ${MASTER_PORT} \
./aicb.py --frame=DeepSpeed --stage=$stage \
--world_size=$((NNODES*8)) --global_batch=$global_batch --epoch_num=$epoch_num \
--num_layers=$num_layers --hidden_size=$hidden_size \
--ffn_hidden_size=$ffn_hidden_size --num_attention_heads=$attention_heads \
--reduce_bucket_size=$bucket_size --allgather_bucket_size=$bucket_size

# Megatron 示例
torchrun \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--nproc_per_node gpu \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
./aicb.py --frame=Megatron --world_size=$((WORLD_SIZE*8)) --tensor_model_parallel_size=$tensor_model_parallel_size \
  --micro_batch=$batch_size --global_batch=$((WORLD_SIZE*8*batch_size/tensor_model_parallel_size)) --epoch_num=$epoch_num --swiglu \
  --num_layers=$num_layers --hidden_size=$hidden_size --ffn_hidden_size=$ffn_hidden_size --num_attention_heads=$num_attention_heads \
  $sp_enable --seq_len=$seq_len --vocab_size=$vocab_size --aiob_enable=$enable 

# MoE 示例
torchrun \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--nproc_per_node gpu \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
./aicb.py --frame=Megatron --world_size=$((WORLD_SIZE*8)) --tensor_model_parallel_size=$tensor_model_parallel_size --expert_model_parallel_size=$expert_model_parallel_size \
--moe_enable=$moe_enable --num_experts=$num_experts --moe_router_topk=$moe_router_topk --moe_grouped_gemm=$moe_grouped_gemm \
  --micro_batch=$batch_size --global_batch=$((WORLD_SIZE*8*batch_size/tensor_model_parallel_size)) --epoch_num=$epoch_num --swiglu \
  --num_layers=$num_layers --hidden_size=$hidden_size --ffn_hidden_size=$ffn_hidden_size --num_attention_heads=$num_attention_heads \
  $sp_enable --seq_len=$seq_len --vocab_size=$vocab_size --aiob_enable=$enable 
```

### 生成工作负载
在AICB中，工作负载的生成分为两种类型：一种是生成可用于仿真和分析的工作负载，另一种是生成包含各种通信和计算详细信息的通用工作负载。
通过提供模型的详细参数，您可以生成一个包含各种通信和计算详细信息的通用工作负载。通用工作负载允许您分析模型在不同参数下的计算和通信性能，并自定义工作负载文件以进行运行时调优和优化。这可以通过以下文件实现：
[generate_deepspeed_stage3_workload](../workload_generator/generate_deepspeed_stage3_workload.py),[generate_deepspeed_stage1_2_workload](../workload_generator/generate_deepspeed_stage1_2_workload.py),[generate_megatron_workload](../workload_generator/generate_megatron_workload.py)

以下是一个示例：
```bash
python -m workload_generator.AIOB_simAI_workload_generator \
  --model_name GPT-13B --frame=Megatron \
  --world_size=16 --tensor_model_parallel_size=2 --pipeline_model_parallel=1 --global_batch=16 \
  --micro_batch=1   --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --epoch_num=1 \
  --use-distributed-optimizer --num_attention_heads=40 \
  --aiob_enable --use_flash_attn --swiglu 
```

#### 工作负载文件
生成的工作负载文件保存在`results/mocked_workload`目录中。
以下是对生成的工作负载文件的说明：
![扩展图](../images/tutorial_5.png)

### 创建自定义模型
AICB具有出色的可扩展性。除了支持GPT和LLaMA系列模型外，它还允许为自定义模型架构创建工作负载。这种灵活性意味着您可以调整AICB以生成和测试各种模型的通信和计算模式，而不仅仅是预配置选项，使其成为基准测试和优化多样化AI训练框架的宝贵工具。
可以使用MockedParam和MockedModel基类构建自定义模型。有关具体实现细节，您可以参考现有的MockedMegatron和MockedDeepSpeed实现。

以下是一个DeepSpeed Llama模型的示例：
```
DeepspeedForCausalLM
      |
      v
+----------------------+
| Linear: embed_tokens |
+----------------------+
      |
      v
+--------------------------------------+
|            DeepspeedModel            |
| +----------------------------------+ |
| | Linear: embed_tokens             | |
| +----------------------------------+ |
| | DeepspeedDecoderLayer x N        | |
| +----------------------------------+ |
| | Linear: norm                     | |
+--------------------------------------+
      |
      v
+--------------------------------------------------+
|              DeepspeedDecoderLayer               |
| +----------------------------------------------+ |
| | Linear: input_layernorm                      | |
| +----------------------------------------------+ |
| | DeepspeedAttention                           | |
| |  +----------------------------------------+  | |
| |  | Linear: q_proj                         |  | |
| |  | Linear: k_proj                         |  | |
| |  | Linear: v_proj                         |  | |
| |  | Linear: o_proj                         |  | |
| |  +----------------------------------------+  | |
| +----------------------------------------------+ |
| | Linear: post_attn_norm                      | |
| +----------------------------------------------+ |
| | DeepspeedMLP                                | |
| |  +----------------------------------------+  | |
| |  | Linear: gate_proj                       |  | |
| |  | Linear: down_proj                       |  | |
| |  | Linear: up_proj                         |  | |
| |  +----------------------------------------+  | |
+--------------------------------------------------+
      |
      v
+----------------------+
| Linear: lm_head      |
+----------------------+

```

除了模型之外，您还需要模拟整个训练过程。所有框架的训练过程被抽象为以下步骤：`init, forward, backward, step`（不包括流水线并行）。我们需要进一步阐述每个步骤中发生的通信以及工作负载项。
在代码中，每个工作负载主要由三个组件组成：
1. 通信信息：
   主要包括与集合通信活动相关的信息，如`comm_type, comm_group, comm_group_size, msg_size`。
2. 附加信息：
   包括补充信息，例如广播通信的源节点和计算操作的时间。
3. 运行时信息：
   主要包括运行时特定的详细信息，如`elapsed_time, algo_bw, bus_bw`，表示集合通信活动的实际性能和状态。
以下是一个简短的训练过程和工作负载项示例：
```python
trainer.init()
for _ in range(epoch_num):
    if pipeline_model_parallel > 1:
        trainer.with_pipeline_forward_backward()
    else:
        for _ in range(num_microbatches):
            trainer.forward()
            trainer.backward()
    trainer.step()
workload.append({
    "operation": "init",             # 对应于init, forward, backward, step之一
    "comm_type": CommType.broadcast, # 通信类型
    "call_func": "_broadcast_model", # 源代码中调用的函数
    "msg_size": param.msg_size(),    # 通信消息的大小
    "comm_group": CommGroup.dp_group,# 通信组
    "src": 0,                        # 可选：仅用于广播，指定源节点
    "additional": send_next          # 可选：指定流水线并行中的相应操作
})
```

# 故障排除与常见问题
暂无