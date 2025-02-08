# 访问 AICB
您可以通过以下链接访问 **SimAI** 工具的完整套件：
- [**SimAI@github**](https://github.com/aliyun/SimAI)

您可以通过以下链接访问 AICB：
- [**AICB@github**](https://github.com/aliyun/aicb)
- [**AICB@gitee**](https://gitee.com/ali-ais-hpn/aicb)

欢迎加入 SimAI 社区聊天群，左侧为钉钉群，右侧为微信群。

<div style="display: flex; justify-content: flex-start; align-items: center; gap: 20px; margin-left: 20px;">
    <img src="./images/simai_dingtalk.jpg" alt="SimAI DingTalk" style="width: 300px; height: auto;">
    <img src="./images/simai_wechat.jpg" alt="SimAI WeChat" style="width: 300px; height: auto;">
</div>

<br/>

# 最新动态
[2024/9] AICB 版本 1.1 发布。
此版本包含以下更新：

### 功能新增
1. 增加了结果可视化功能，支持在物理集群运行后显示结果，并支持生成的工作负载文件的可视化。详情请参阅 Readme。
2. 优化了通信组划分方法，增强了可扩展性。
3. 增加了对 MOE `group_gemm` 的 AIOB 计算模式的支持。
对 `run_in_cluster` 脚本进行了一些优化。

### Bug 修复
1. 修复了日志中部分 BusBw 计算错误的问题。
2. 修复了多机运行时 AIOB 异常计算时间的问题。
3. 修复了启用 `computation_enable` 时 comm_log 统计异常的问题。
4. 修复了 `run_suite` 脚本潜在的挂起问题。
5. 修复了使用 `tp=1` 和 `ep=1` 时生成 simAI 工作负载描述文件的错误。
6. 修复了与 MOE 相关的部分消息大小错误。

# 目录

- [访问 AICB](#访问-aicb)
- [最新动态](#最新动态)
- [目录](#目录)
- [AICB 概述](#aicb-概述)
  - [简介](#简介)
  - [AICB 中的基准测试套件](#aicb-中的基准测试套件)
- [环境搭建](#环境搭建)
- [使用方法](#使用方法)
  - [在物理 GPU 集群上运行](#在物理-gpu-集群上运行)
    - [需要设置的基本参数](#需要设置的基本参数)
    - [运行整个基准测试套件](#运行整个基准测试套件)
    - [运行 Megatron 的工作负载](#运行-megatron-的工作负载)
    - [运行 MOE 的工作负载](#运行-moe-的工作负载)
    - [运行 DeepSpeed 的工作负载](#运行-deepspeed-的工作负载)
    - [嵌入工作负载中的计算模式](#嵌入工作负载中的计算模式)
  - [为仿真（SimAI）生成工作负载](#为仿真-simai-生成工作负载)
    - [为整个基准测试套件生成工作负载描述文件](#为整个基准测试套件生成工作负载描述文件)
    - [为 Megatron 生成工作负载描述文件](#为-megatron-生成工作负载描述文件)
    - [为 MOE 生成工作负载描述文件](#为-moe-生成工作负载描述文件)
    - [为 DeepSpeed 生成工作负载描述文件](#为-deepspeed-生成工作负载描述文件)
  - [使用自定义参数运行 AICB](#使用自定义参数运行-aicb)
    - [在物理 GPU 集群上运行自定义工作负载](#在物理-gpu-集群上运行自定义工作负载)
    - [生成自定义工作负载描述文件](#生成自定义工作负载描述文件)
  - [结果可视化](#结果可视化)
- [教程](#教程)
- [使用 AICB 的项目](#使用-aicb-的项目)

# AICB 概述
## 简介
AICB（Artificial Intelligence Communication Benchmark）是一个新颖的基准测试套件，用于从新兴的训练和推理应用的角度评估真实和模拟 GPU 集群的通信系统。与现有的网络基准测试不同，AICB 设计用于生成与实际应用对齐的精确通信工作负载模式。以大型语言模型（LLM）训练为例，工作负载会因模型、并行框架和集体通信库的复杂组合而变化。总的来说，适合使用 AICB 的场景包括但不限于：1）GPU 集群通信系统的基准测试和调优；2）特定应用场景通信模式的研究和分析；3）需要良好描述工作负载的工具（如模拟器）。

## AICB 中的基准测试套件
许多参数会影响通信和计算模式，包括：1）模型参数（如 hidden_size、num_layers、seq_len 等）；2）框架参数（如 world size、并行策略（TP、PP、DP、SP）、zero level、reduce_bucket_size/allgather_bucket_size 等）。为了通用性，我们通过最小的基准测试集覆盖这些典型设置，而不是遍历所有组合。以下是基准测试套件的列表。
**用户可以直接运行 AICB 中选定的所有工作负载，也可以选择运行部分工作负载，甚至生成自己的工作负载。**
更多详细信息，请参考 [AICB_workload spec v1.1](workload/Workload_spec_v1.1.csv)。

| id  | 名称          | 序列长度 | 框架     | TP  | DP                    | PP  | SP     | 专家并行数 | 专家数 | Zero 级别 |
|:---:|:-------------:|:-------:|:-------:|:---:|:---------------------:|:---:|:------:|:--------:|:----:|:--------:|
|  1  | LLaMA_7B      |  2048   | Megatron |  1  |  world_size/(PP*TP)   |  1  |   -    |    -     |  -   |    -     |
|  2  | GPT_13B       |  2048   | Megatron |  2  |  world_size/(PP*TP)   |  1  | enable |    -     |  -   |    -     |
|  3  | GPT_22B       |  2048   | Megatron |  4  |  world_size/(PP*TP)   |  1  |   -    |    -     |  -   |    -     |
|  4  | LLaMA_65B     |  4096   | Megatron |  8  |  world_size/(PP*TP)   |  2  | enable |    -     |  -   |    -     |
|  5  | GPT_175B      |  2048   | Megatron |  8  |  world_size/(PP*TP)   |  8  | enable |    -     |  -   |    -     |
|  6  | GPT_175B      |  2048   | Megatron |  8  |  world_size/(PP*TP)   |  8  | disable|    -     |  -   |    -     |
|  7  | Llama3_405B   |  8192   | Megatron |  8  |  world_size/(PP*TP)   |  16 | enable |    -     |  -   |    -     |
|  8  | LLaMA_7B      |  4096   | Deepspeed|  1  |      world_size       |  1  |   -    |    -     |  -   |    2     |
|  9  | LLaMA_65B     |  4096   | Deepspeed|  1  |      world_size       |  1  |   -    |    -     |  -   |    3     |
| 10  | Mistral_8*7B  |  2048   | Megatron |  2  |  world_size/(PP*TP)   |  1  | enable |    8     |  8   |    -     |

# 环境搭建
您可以按照以下步骤快速搭建环境并运行 AICB。

1. 从源代码安装

    a. 为了启动实际通信任务，确保运行时环境已安装所有必要的依赖项，例如 CUDA 和 [PyTorch](https://pytorch.org)。有关具体用法示例，请参见 [物理执行](#物理执行)。

    b. 为了生成大模型并行框架训练的工作负载流量模式，可以使用仅 CPU 的环境。有关具体用法示例，请参见 [生成工作负载](#生成工作负载)。

2. 从 deb 包安装（适用于 Ubuntu 系统）

    当前，您可以在 NV 构建的 NGC 容器 [NGC's PyTorch 容器](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) 上安装 deb 包以开始运行 AICB。
    ```bash
    docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
    docker run --gpus all -it --rm -v /path/to/AICBench:/workspace/AICBench nvcr.io/nvidia/pytorch:xx.xx-py3
    dpkg -i /download/AICB_v1.0.deb 
    sh megatron_workload_with_aiob.sh -m 7
    ```

3. 使用 Dockerfile 构建 Docker 镜像

    您可以通过 Dockerfile 快速启动 Docker 容器实例：
    ```bash
    docker build -t image:latest .
    docker run --gpus all -it --rm image:latest 
    ```

# 使用方法
安装完成后，我们为 AICB 提供了三种主要使用场景：
1. [在物理 GPU 集群上运行](#在物理-gpu-集群上运行)
2. [为仿真生成工作负载描述文件](#为仿真-simai-生成工作负载)
3. [使用自定义参数](#使用自定义参数)。

我们提供了一个包含所有细节的教程，请参阅 [教程](training/tutorial.md)。

## 在物理 GPU 集群上运行
对于在物理机器上运行 AICB，我们提供了 [脚本](scripts/megatron_gpt.sh) 用于快速启动以及 [方法](aicb.py) 用于执行自定义案例。

### 需要设置的基本参数
在物理机器上运行时，需要额外配置 PyTorch 所需的环境变量。
```bash
--nnodes                  节点数量: $WORLD_SIZE
--node_rank               节点排名: $RANK
--nproc_per_node          每个节点的 GPU 数量: $NUM_GPUS
--master_addr             主地址: $MASTER_ADDR
--master_port             主端口: $MASTER_PORT
```

### 运行整个基准测试套件
您可以通过 [run_suites](run_suites.py) 脚本直接执行我们在 AICB 工作负载规范 v1.0 中提供的所有测试用例。该脚本确保涵盖所有并行框架，帮助您高效验证和分析各种工作负载的性能和行为。

### 运行 Megatron 的工作负载
对于 `Megatron 并行框架`，您可以使用 scripts/megatron_gpt.sh 脚本文件快速启动。
```bash
sh scripts/megatron_gpt.sh \
--nnodes 1 --node_rank 0 --nproc_per_node 8 --master_addr localhost --master_port 29500 \
-m 7 --world_size 8 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 16  \
--micro_batch 1 --seq_length 2048 --swiglu --use_flash_attn --aiob_enable
```

### 运行 MOE 的工作负载
对于 `MOE`，您可以使用 [scripts/megatron_gpt.sh](scripts/megatron_gpt.sh) 脚本文件快速启动。
```bash
sh scripts/megatron_gpt.sh \
--nnodes 1 --node_rank 0 --nproc_per_node 8 --master_addr localhost --master_port 29500 \
-m moe --world_size 8 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
--moe_enable --expert_model_parallel_size 1  \
--frame Megatron --global_batch 16  \
--num_experts 4 --moe_router_topk 2 \
--micro_batch 1  --sp --grouped_gemm --aiob_enable --swiglu --use_flash_attn 
```

### 运行 DeepSpeed 的工作负载
对于 `DeepSpeed` 并行框架，您可以使用 [scripts/deepspeed_llama.sh](scripts/deepspeed_llama.sh) 脚本文件快速启动。目前，DeepSpeed 框架不支持 `--aiob_enable` 或 `--comp_filepath`，但您可以选择使用固定计算时间（请参阅 [教程](training/tutorial.md)）。
```bash
sh scripts/deepspeed_llama.sh \
--zero_stage 3 -m 65 --epoch_num 100 \
--reduce_bucket_size=1000000000 --allgather_bucket_size=500000000 \
--param_persistence_threshold=1000000 \
```

### 嵌入工作负载中的计算模式
为了反映现实世界中同时包含计算和通信的工作负载，我们开发了一个子模块 AIOB，用于生成计算模式。在 AICB 中，我们可以通过启用 AIOB 将计算时间嵌入到工作负载中。

对于 Megatron 并行框架，`--aiob_enable` 选项允许捕获实际模型中每个操作的计算时间。如果我们未设置 `--aiob_enable`，则只能应用固定计算时间。（请参阅 [教程](training/tutorial.md)）

* 使用 AIOB 生成的计算时间运行工作负载。运行后，我们可以在 [results/aiob_outputs](results/aiob_outputs) 目录中获得一个额外的计算描述文件，描述主要计算内核的计算时间。请注意，计算时间是通过在特定 GPU 上执行计算内核获得的。以下命令不仅生成计算描述文件，还运行实际 GPU 集群上的工作负载。
```bash
sh scripts/megatron_gpt.sh \
-m 7 --world_size 8 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 16  \
--micro_batch 1 --seq_length 2048 \
--swiglu --use_flash_attn  --aiob_enable 
```
* 使用现有计算描述文件运行工作负载。
用户可以定义自己的计算时间或直接使用我们提供的文件。通过使用 `--comp_filepath` 选项指定计算描述文件，您可以在物理机器上运行工作负载之前嵌入计算时间。
```bash
sh scripts/megatron_gpt.sh \
-m 7 --world_size 8 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 16  --micro_batch 1 \
--seq_length 2048 --swiglu --use_flash_attn  \
--aiob_enable  \
--comp_filepath workload/aiob_inputs/Example.txt
```

```markdown
## 为仿真（SimAI）生成工作负载
除了在 GPU 集群上运行 AICB 外，AICB 还可以生成可用于仿真或进一步分析的工作负载描述文件。在此版本中，我们提供了 [脚本](scripts/megatron_workload_with_aiob.sh) 用于快速生成 SimAI 的工作负载。

### 为整个基准测试套件生成工作负载描述文件
您可以使用 [generate_suite]() 根据我们的 AICB 工作负载规范 v1.0 生成所有工作负载描述文件。一旦这些文件生成完成，您可以使用 SimAI 执行它们以测试和分析各种场景。

### 为 Megatron 生成工作负载描述文件
在这里，您可以使用脚本 [scripts/megatron_workload.sh](scripts/megatron_workload_with_aiob.sh) 和参数 `--model_size`（7/13/22/175/moe）生成相应的工作负载描述文件。对于模型的计算部分，您可以选择通过使用 `--aiob_enable` 选项启用 AIOB。如果未使用 AIOB，则默认情况下工作负载将填充固定计算时间。
* 使用 AIOB 生成的计算时间生成工作负载描述文件。
```bash
sh ./scripts/megatron_workload_with_aiob.sh \
-m 7 --world_size 4096 \
--tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 \
--swiglu --use_flash_attn  --aiob_enable
```
* 使用现有计算描述文件生成工作负载描述文件。
```bash
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 4096 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt
```

### 为 MOE 生成工作负载描述文件
对于 MOE，您也可以使用 [scripts/megatron_workload_with_aiob.sh](scripts/workload_megatron.sh) 生成相应模型的工作负载文件。
```bash
sh scripts/megatron_workload_with_aiob.sh \
-m moe --world_size 512 --tensor_model_parallel_size 2 --pipeline_model_parallel 1 --sp  --ep 16 \
--num_experts 64 --moe_router_topk 2 --moe_grouped_gemm --moe_enable  \
--frame Megatron --global_batch 1024  \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable 
```

### 为 DeepSpeed 生成工作负载描述文件
对于 `DeepSpeed` 并行框架，您可以使用 [scripts/workload_deepspeed.sh](scripts/workload_deepspeed.sh) 生成相应的工作负载描述文件。
```bash
sh ./scripts/workload_deepspeed.sh -m 7 
```

## 使用自定义参数运行 AICB
除了快速启动外，您还可以详细自定义模型参数，以便在物理集群上运行或生成用于仿真和分析所需的工作负载。有关更详细的参数描述和更多示例，请参阅 [教程](training/tutorial.md)。

### 在物理 GPU 集群上运行自定义工作负载
当前运行自定义案例的入口文件是 [aicb.py](aicb.py)。通过使用此文件，您可以灵活选择更多参数进行调优。
```bash
# Megatron 示例
torchrun \
--nnodes $WORLD_SIZE \
--node_rank $RANK \
--nproc_per_node gpu \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT \
./aicb.py --frame=Megatron --world_size=$((WORLD_SIZE*8)) --tensor_model_parallel_size=$tensor_model_parallel_size \
  --micro_batch=$batch_size --global_batch=$((WORLD_SIZE*8*batch_size/tensor_model_parallel_size)) --epoch_num=$epoch_num \
  --num_layers=$num_layers --hidden_size=$hidden_size --ffn_hidden_size=$ffn_hidden_size --num_attention_heads=$num_attention_heads \
  $sp_enable --seq_len=$seq_len --vocab_size=$vocab_size --aiob_enable=$enable 
```

### 生成自定义工作负载描述文件
同样，在生成工作负载时，您也可以自定义模型训练参数，并修改生成的文件以生成自己的仿真工作负载文件。这可以通过以下文件实现：
[生成自定义描述文件](workload_generator/AIOB_simAI_workload_generator.py)

以下是一个示例：
```bash
python -m workload_generator.AIOB_simAI_workload_generator \
--world_size=32  --global_batch=64 --micro_batch=1 \
--num_layers=8 --num_attention_heads=176 --hidden_size=5120   \
--tensor_model_parallel_size=2 --seq_length=4096 --swiglu --ffn_hidden_size=16384  \
--moe_router_topk=4  --enable_sequence_parallel --expert_model_parallel_size=16 \
--num_experts=64 --moe_grouped_gemm --moe_enable --num_experts=4
```

## 结果可视化

本节介绍结果可视化功能。

支持的格式：`.csv` 文件支持可视化，包括来自物理集群运行的结果和工作负载文件。

用法：
Post-Run 和生成的工作负载文件都可以进行可视化。您可以使用 [visualize_script](visualize/generate.py) 来可视化结果。
以下是工作负载文件的示例：
```bash
python -m visualize.generate ./local_megatron_workload.csv only_workload
```
Post-Run 结果可视化的示例：
```bash
python -m visualize.generate ./megatron_postrun.csv
```
输出结果位于 `results/visual_output` 目录中。您可以通过该目录中的 `example.html` 文件查看输出样式。生成的可视化文件是一个 HTML 文件，可以在浏览器中打开并查看，如下所示。
![Scaling Graph](./images/readme_01.png)

结果包含以下几个部分：
- 通信结果饼图：显示在给定训练超参数下各种集体通信的数量和比例。
- 通信类型散点图：显示每种通信类型的通信消息大小和计数。对于实际物理集群运行的结果，还会显示相应的 BusBw。
- 集体通信中消息大小的 CDF 图：展示不同类型集体通信的消息大小分布。
- 通信组散点图：显示不同模型训练通信组的通信消息大小和计数。对于实际物理集群运行的结果，还会显示相应的 BusBw。
- 计算与通信时间轴（仅支持物理集群运行）：显示 AICB 运行期间计算和通信事件的时间轴。时间轴可以拖动以观察特定的计算和通信事件。
- 总体计算与通信比例（仅支持物理集群运行）：显示 AICB 运行期间总时间中计算和通信所占的比例。

# 教程
我们提供了一个教程，帮助用户快速上手 AICB。[教程](./training/tutorial.md)

# 使用 AICB 的项目
以下是我们直接使用 AICB 的一些项目：
* AICB 是由阿里云主导的 SimAI 项目的一部分。使用 AICB 的研究人员可以引用我们的论文 "SimAI: Unifying Architecture Design and Performance Tunning for Large-Scale Large Language Model Training with Scalability and Precision" (NSDI’25)。
```