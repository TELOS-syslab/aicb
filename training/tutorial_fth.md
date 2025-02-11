# Introduction
AI Communication Benchmark is a specialized communication benchmarking suite designed for artificial intelligence (AI) scenarios, primarily used to evaluate the performance of communication stacks. This suite not only provides detailed performance metrics but also assists developers in quickly identifying and diagnosing potential performance bottlenecks and issues within the communication stack. By simulating real-world communication traffic patterns during AI training and inference processes, this benchmark accurately reflects the communication stack's actual performance under conditions of high concurrency and large data transfer volumes. Whether it's inter-node communication in distributed computing or data synchronization in large-scale model training, this benchmarking suite offers effective performance evaluation and optimization recommendations to help users enhance overall system efficiency and stability.
简介
AI通信基准测试（AI Communication Benchmark）是一个专为人工智能（AI）场景设计的通信基准测试套件，主要用于评估通信栈的性能。
该套件不仅提供详细的性能指标，还帮助开发者快速识别和诊断通信栈中潜在的性能瓶颈和问题。
通过模拟AI训练和推理过程中的实际通信流量模式，该基准测试能够准确反映在高并发和大数据传输量条件下的通信栈实际性能。
无论是分布式计算中的节点间通信，还是大规模模型训练中的数据同步，该基准测试套件都提供了有效的性能评估和优化建议，以帮助用户提升整体系统的效率和稳定性。


# Environment Setup
Before setting up the environment, first pull the code repository to your local machine and then proceed with the environment configuration:
环境搭建
在搭建环境之前，首先将代码仓库拉取到本地，然后进行环境配置：

```
git clone https://github.com/aliyun/aicb.git
```
For the environment, if you are only generating workloads, no additional dependencies are needed. 
However, other functionalities require dependencies such as PyTorch, CUDA, NCCL, and NVIDIA APEX. 
Therefore, you can set up an appropriate runtime environment either by configuring your local environment or using Docker.
对于环境，如果您只是生成工作负载，则不需要额外的依赖项。
然而，其他功能需要依赖如PyTorch、CUDA、NCCL和NVIDIA APEX等库。因此，您可以通过配置本地环境或使用Docker来设置适当的运行环境。


## Setting the environment using a Dockerfile. 使用Dockerfile设置环境
```
docker build -t aicb:v0.0.1 .
docker run --gpus all --net host --shm-size 16g -it --rm aicb:v0.0.1 
```
## Setting the Environment Locally 本地环境搭建
For a local environment, you will need Python >= 3.8, CUDA Version >= 11.8, PyTorch >= 2.0.0, and NVIDIA APEX.
对于本地环境，您需要 Python >= 3.8、CUDA 版本 >= 11.8、PyTorch >= 2.0.0 和 NVIDIA APEX。

## Using Official Docker 使用官方Docker
You can also create the required Docker environment using [NGC's PyTorch container ](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch), where pytorch:xx.xx should be >= pytorch:23.08.
您还可以使用 NGC的PyTorch容器 创建所需的Docker环境，其中 pytorch:xx.xx 应该 >= pytorch:23.08。

```
docker pull nvcr.io/nvidia/pytorch:xx.xx-py3
docker run --gpus all -it --rm -v /path/to/AICBench:/workspace/AICBench nvcr.io/nvidia/pytorch:xx.xx-py3
```

# Basic Usage 
## Physical Execution
基本用法
物理执行

When running on a physical machine, additional configuration for PyTorch-related environment variables is required. 
This can be done by explicitly specifying them in a script or by adding the environment variables directly.
The following table lists the required environment variables:
在物理机上运行时，需要对PyTorch相关的环境变量进行额外配置。
这可以通过在脚本中显式指定它们或将环境变量直接添加到系统中完成。
下表列出了所需的环境变量：

| Parameter Name  | Description                   |
|-----------------|-------------------------------|
| nnodes          | Number of nodes               |
| node_rank       | Rank number of the node       |
| nproc_per_node  | Number of GPUs per node       |
| master_addr     | Address of the master node    |


参数名称	描述
nnodes	节点数量
node_rank	节点的排名编号
nproc_per_node	每个节点的GPU数量
master_addr	主节点地址



### Quick start for single-node execution 单节点执行快速入门
The script for running AICB on a physical machine is：
在物理机上运行AICB的脚本是：
[/scripts/megatron_workload_with_aiob.sh](../scripts/megatron_workload_with_aiob.sh)

We provide four pre-existing models (7/13/22/175/)B and moe to quickly launch and run on a physical machine, which can be specified using the parameter `--model_size`. 
Additionally, the Megatron parallel framework supports enabling the aiob_enable option to obtain the computation time for each operation of the actual model. 
Without using aiob, only fixed waiting times can be filled. 
Alternatively, when AIOB is enabled, you can specify `--comp_filepath` to fill in the corresponding computation time.
我们提供了四种预置模型（7/13/22/175B）和MoE模型，可以快速启动并运行在物理机上，可以通过参数--model_size指定。
此外，Megatron并行框架支持启用aiob_enable选项，以获取实际模型每个操作的计算时间。
如果不使用aiob，则只能填充固定等待时间。
或者，当启用AIOB时，您可以指定--comp_filepath以填充相应的计算时间。

Below is an example of generating a Workload with a model size of 13B, tp 8, pp 1, a total GPU count of 8, gbs 2, mbs 1, sequence length of 4096, with flash_attn and swiglu enabled, and using AIOB to obtain the computation time for each operation of the actual model.
 以下是一个生成模型大小为13B、tp 8、pp 1、总GPU数为8、gbs 2、mbs 1、序列长度为4096，并启用flash_attn和swiglu的Workload示例，同时使用AIOB获取实际模型每个操作的计算时间。

``` bash
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
### Quick start for multi-node execution 多节点执行快速入门
The script used for multi-node execution is：[run_in_cluster.py](../scripts/run_in_cluster.py)
用于多节点执行的脚本是：run_in_cluster.py

Steps： 步骤：
1. First, install the batch distribution commands (such as pssh and pscp).
首先安装批量分发命令（如pssh和pscp）。

2. Edit the `iplist` file of the cluster to be used, adding an accessible IP address of each machine per line to the iplist.
编辑要使用的集群的iplist文件，每行添加一台机器的可访问IP地址。

3. Modify [run_in_cluster.py](../scripts/run_in_cluster.py) to specify the image name and paths to `iplist` file and AICB home directory. Please refer to the doc of [run_in_cluster.py](../scripts/run_in_cluster.py) for more details.
修改run_in_cluster.py，指定镜像名称以及iplist文件和AICB主目录的路径。更多细节请参考run_in_cluster.py的文档。 

4. Modify [run_suites.py](../run_suites.py) to select the workload to run (default: no workload).
修改run_suites.py，选择要运行的工作负载（默认：无工作负载）。

5. Copy the `iplist` and AICB source code to each machine (e.g., using pscp).
将iplist和AICB源代码复制到每台机器（例如，使用pscp）。

4. Run the command just like this: `pssh -i -h /path/to/iplist -o out -e err -t 0 "cd /path/to/aicb && python run_in_cluster.py"`. Remember to replace `/path/to/iplist` and `/path/to/aicb` with the actual path on your machine.
运行命令如下：pssh -i -h /path/to/iplist -o out -e err -t 0 "cd /path/to/aicb && python run_in_cluster.py"。记得将/path/to/iplist和/path/to/aicb替换为您机器上的实际路径。

The specific command to be run on each machine can be modified in the highlighted section:
每台机器上运行的具体命令可以在突出显示的部分修改：

![Scaling Graph](../images/tutorial_7.png)
### Logs and Results 日志与结果
After each communication is completed, the program will print the relevant logs for this communication. The output format is as follows and contains almost all the information about the communication operation:
每次通信完成后，程序将打印此次通信的相关日志。输出格式如下，几乎包含所有通信操作的信息：

* communication type
* communication group
* Message size
* Communication execution time
* [Throughput](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)

通信类型
通信组
消息大小
通信执行时间
吞吐量

![Scaling Graph](../images/tutorial_1.png)

After all communications are completed, information about the use case will be summarized.
所有通信完成后，将汇总关于用例的信息：

* This includes the overall runtime and an analysis of the time for each iteration, allowing for a clear view of whether each iteration runs normally and if there are any jitters.
包括总体运行时间和每次迭代的时间分析，可以清楚地查看每次迭代是否正常运行以及是否存在抖动。

![Scaling Graph](../images/tutorial_2.png)

* Time for each communication type 每种通信类型的时间
First, distinguish between the model training phases: the init phase and the train phase. 
Then, summarize the collective communications performed in each phase, including their corresponding message sizes, frequencies, and specific average latencies, maximum and minimum values, etc. 
This helps to pinpoint which type of collective communication operation in which message segment is causing anomalies, facilitating further investigation and troubleshooting.
首先区分模型训练阶段：初始化阶段和训练阶段。
然后总结每个阶段中执行的集合通信，包括其对应的消息大小、频率以及具体的平均延迟、最大值和最小值等。
这有助于确定哪个消息段的集合通信操作导致了异常，便于进一步调查和故障排除。

![Scaling Graph](../images/tutorial_3.png)

#### File outputs 文件输出
The file outputs include two different types of files: .csv file. 文件输出包括两种不同类型的文件：.csv文件。
1. The CSV files are saved in: CSV文件保存在：
`results/comm_logs/megatron_gpt_13B_8n_log.csv`,And you can also see the execution time, the execution phase, as well as the algorithmic bandwidth (algbw) and bus bandwidth (busbw) belonging to different comm_group and different comm_type.It also includes the computation time for each part of the model and the computation phase it belongs to.
results/comm_logs/megatron_gpt_13B_8n_log.csv，您还可以看到执行时间、执行阶段以及属于不同comm_group和comm_type的算法带宽（algbw）和总线带宽（busbw）。它还包括模型每个部分的计算时间和所属的计算阶段。

![Scaling Graph](../images/tutorial_4.png)

Inaddition to the aforementioned details, a .csv file is provided for detailed analysis of the results. Here’s how to work with it:
除了上述详细信息外，还提供了一个.csv文件用于详细分析结果。以下是使用方法：

    1. Reading _workload.csv Log: 读取_workload.csv日志：
      * You can read the _workload.csv log file by invoking log_analyzer.log.Workload.load(filename).
       可以通过调用log_analyzer.log.Workload.load(filename)读取_workload.csv日志文件。

      * This will return Workload and args. 这将返回Workload和args。
        * args contains the parameters used for training input. args 包含用于训练输入的参数。
        * Workload consists of the generated intermediate results. Workload由生成的中间结果组成。

    2. Reading _log.csv Log: 读取_log.csv日志：
    * You can read the _log.csv log file by invoking log_analyzer.log.Log.load(filename). 
    可以通过调用log_analyzer.log.Log.load(filename)读取_log.csv日志文件。
    
    * This will return a Log object, containing: 这将返回一个Log对象，包含：
      * comm_logs: List[LogItem]: This is a list of all generated logs. 这是所有生成日志的列表。
      
      * epoch_times: List[int]: This lists the time taken for each iteration. The first iteration typically represents initialization, which might show different communication behavior compared to subsequent iterations, potentially leading to differences in time. 这是每次迭代所花费的时间列表。第一次迭代通常表示初始化，可能会显示与后续迭代不同的通信行为，可能导致时间差异。

      * comm_log_each_epoch: List[List[LogItem]]: This is a list where each item corresponds to the communication logs for each iteration. If one iteration has a significantly different time compared to others, you can analyze this specific iteration to identify the communication causing the discrepancy.这是一个列表，其中每个项目对应每次迭代的通信日志。如果某次迭代的时间与其他迭代显著不同，您可以分析此次迭代以识别导致差异的通信。

By leveraging these log files and parsing methods, you can perform a thorough and detailed analysis of the training process, identifying any abnormalities or areas for optimization.
通过利用这些日志文件和解析方法，您可以对训练过程进行全面而详细的分析，识别任何异常或优化领域。




## Generate Workload for Simulation(SimAI) 为仿真生成工作负载（SimAI）
### Quick start 快速入门
AICB's script for generating Workload is: `./scripts/megatron_workload_with_aiob.sh`
AICB生成工作负载的脚本是：./scripts/megatron_workload_with_aiob.sh

We provide four pre-existing models (7/13/22/175)B to quickly generate the corresponding Workload, which can be specified using the parameter `--model_size`.
The computation part of the model can be selected to use aiob via--aiob_enableWhen not using aiob, the default fixed time is used to fill the Workload.
When `--aiob_enable` is enabled, if `--comp_filepath` is not specified, the current GPU's computation time will be used to fill the Workload
 我们提供了四种预置模型（7/13/22/175B），可以快速生成相应的工作负载，可以通过参数--model_size指定。
 模型的计算部分可以选择通过--aiob_enable使用aiob。当不使用aiob时，默认使用固定时间填充工作负载。
 当启用--aiob_enable时，如果不指定--comp_filepath，则使用当前GPU的计算时间填充工作负载。

Below is an example of generating a Workload with a model size of 7B, tp 4, pp 1, a total GPU count of 4096, gbs 8192, mbs 1, sequence length of 4096, with flash_attn, swiglu, and aiob enabled, and reading Example.txt as the computation time.
 以下是一个生成模型大小为7B、tp 4、pp 1、总GPU数为4096、gbs 8192、mbs 1、序列长度为4096，并启用flash_attn、swiglu和aiob，同时读取Example.txt作为计算时间的工作负载示例。

```bash
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 4096 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt
```


sudo docker exec -it fth_simai /bin/bash


```bash fth
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 4096 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable \
--comp_filepath workload/aiob_inputs/Example.txt
```

```bash fth
sh ./scripts/megatron_workload_with_aiob.sh -m 7 \
--world_size 4096 --tensor_model_parallel_size 4 --pipeline_model_parallel 1 \
--frame Megatron --global_batch 8192 \
--micro_batch 1 --seq_length 4096 --swiglu \
--use_flash_attn  --aiob_enable 
```

### Workload
The generated Workload result is saved in:`results/workload/gpt_7B-world_size4096-tp4-pp1-gbs8192-mbs1-seq4096-flash_attn-True.txt`
工作负载
生成的工作负载结果保存在：results/workload/gpt_7B-world_size4096-tp4-pp1-gbs8192-mbs1-seq4096-flash_attn-True.txt

![Scaling Graph](../images/tutorial_6.png)

## Run AICB with customized cases 使用自定义案例运行AICB

In addition to the quick start options, you can also customize the model parameters in detail to run on physical machines or generate the required workloads for simulation and analysis. 
This flexibility allows you to tailor the workloads specifically to your needs, whether you are experimenting with different configurations of large language models, testing various parallel frameworks, or optimizing your runtime environment. 
Customizing parameters provides deeper insights and greater control over the benchmarking and simulation processes, enabling more precise performance tuning and analysis.
除了快速入门选项外，您还可以详细自定义模型参数，以在物理机上运行或生成所需的仿真和分析工作负载。
这种灵活性允许您根据需求定制工作负载，无论您是在尝试不同配置的大语言模型、测试各种并行框架，还是优化运行时环境。
自定义参数提供了对基准测试和仿真过程的更深入洞察和更大控制力，从而实现更精确的性能调优和分析。

### Parameters 参数
The main parameters for AICB are as follows: 

AICB的主要参数如下：

| Category                     | Parameter Name                   | Description                                                                 | 
|------------------------------|-----------------------------------|-----------------------------------------------------------------------------|
| Name                         | frame                             | DeepSpeed/Megatron                                                          |
|                              | model_name                        | Llama/GPT/...                                                               |
|Training Parameters           | world_size                        | Total number of GPUs                                                        |
|                              | global_batch                      | Total batch size for training                                               |
|                              | micro_batch                       | Batch size per model instance (local batch size).                           |
|                              | epoch_num                         | Number of iterations                                                        |
| Model parameters             | model_size                        | Model size (7/13/65/175/270)B and moe                                       |
|                              | num_layers                        | Number of transformer layers.                                               |
|                              | hidden_size                       | Transformer hidden size.                                                    |
|                              | num_attention_heads               | Number of transformer attention heads.                                      |
|                              | seq_length                        | Maximum sequence length to process.                                         |
|                              | vocab_size                        | Size of vocab before EOD or padding.                                        |
|                              | max_position_embeddings           | Maximum number of position embeddings to use.                               |
|                              | ffn_hidden_size                   | Transformer Feed-Forward Network hidden size.                               |
| Megatron parallel parameters | tensor_model_parallel_size        | Degree of tensor model parallelism.                                         |
|                              | pipeline_model_parallel           | Degree of pipeline model parallelism.                                       |
|                              | enable_sequence_parallel          | Enable sequence parallel optimization.                                      |
| Megatron optimization parameters | use_flash_attn                | Use FlashAttention implementation of attention.                             |
|                              | swiglu                            | Use gated linear units and SiLU activation instead of default gelu          |
|                              | openai_gelu                       | Use OpenAI's GeLU implementation.                                           |
|                              | onnx_safe                         | Use workarounds for known problems with Torch ONNX exporter                 |
|                              | squared_relu                      | Use squared relu activation instead of default gelu                         |
|                              | bias_gelu_fusion                  | Enable bias and gelu fusion.                                                |
|                              | gated_linear_unit                 | Enable when use swiglu                                                      |
| MoE                          | expert_model_parallel_size        | Degree of expert model parallelism                                          |
|                              | moe_enable                        | Enable MoE                                                                  |
|                              | num_experts                       | Number of Experts in MoE (None means no MoE)                                |
|                              | moe_router_topk                   | Number of experts to route to for each token.                               |
|                              | moe_grouped_gemm                  | When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel|
| DeepSpeed parameters         | zero_stage,reduce_bucket_size     | choose zero optimizer stage                                                 |
|                              | allgather_bucket_size             | Optimizes communication efficiency and memory usage during all-gather operations For stage 1/2 only                                                          |
|                              | prefetch_bucket_size, param_persistence_threshold, model_persistence_threshold, max_live_parameters | For stage 3 only. Control the number of prefetch parameters. Control the size of all_gather and reduce_scatter |
| Other                        | aiob_enable                       | Enable AIOB to obtain computation time                                      |
|                              | comp_filepath                     | Use aiob_lib to get operation compute time                                  |

### Running on physical GPU clusters 在物理GPU集群上运行
The current entry file for running custom cases is [aicb.py](../aicb.py). By using this file, you can flexibly choose more parameters for tuning.
当前运行自定义案例的入口文件是[aicb.py](../aicb.py)。通过使用此文件，您可以灵活选择更多参数进行调优。

```bash
# DeepSpeed Stage 2 Example
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

# Megatron Example
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

# MoE Example
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
### Generating Workloads
In AICB, the generation of Workloads is divided into two types: one is to generate Workloads that can be used for simulation and analysis, and the other is to generate general-purpose Workloads that contain detailed information about various communication and computation aspects.
By providing the detailed parameters of the model, you can generate a general-purpose Workload that includes detailed information about various communications and computations. The general-purpose workload allows you to analyze the model's computational and communication performance under different parameters, and customize the workload files for run-time tuning and optimization. This can be achieved by using the following files:
生成工作负载
在AICB中，工作负载的生成分为两种类型：一种是生成可用于仿真和分析的工作负载，另一种是生成包含各种通信和计算详细信息的通用工作负载。
通过提供模型的详细参数，您可以生成一个包含各种通信和计算详细信息的通用工作负载。通用工作负载允许您分析模型在不同参数下的计算和通信性能，并自定义工作负载文件以进行运行时调优和优化。这可以通过以下文件实现：

[generate_deepspeed_stage3_workload](../workload_generator/generate_deepspeed_stage3_workload.py),[generate_deepspeed_stage1_2_workload](../workload_generator/generate_deepspeed_stage1_2_workload.py),[generate_megatron_workload](../workload_generator/generate_megatron_workload.py)

Here is an example:
```bash
python -m workload_generator.AIOB_simAI_workload_generator \
  --model_name GPT-13B --frame=Megatron \
  --world_size=16 --tensor_model_parallel_size=2 --pipeline_model_parallel=1 --global_batch=16 \
  --micro_batch=1   --num_layers=40 --seq_length=2048 \
  --hidden_size=5120 --epoch_num=1 \
  --use-distributed-optimizer --num_attention_heads=40 \
  --aiob_enable --use_flash_attn --swiglu 
```
#### Workload Files 工作负载文件
The generated Workload files are saved in the `results/mocked_workload` directory.
Here is an explanation of a the generated Workload file:
生成的工作负载文件保存在`results/mocked_workload`目录中。
以下是对生成的工作负载文件的说明：
![Scaling Graph](../images/tutorial_5.png)

### Creating customized Models 创建自定义模型
AICB offers remarkable extensibility. In addition to supporting GPT and LLaMA series models, it also allows for the creation of workloads for custom model architectures. This flexibility means you can adapt AICB to generate and test communication and computation patterns for a wide variety of models, beyond the pre-configured options, making it an invaluable tool for benchmarking and optimizing diverse AI training frameworks.
AICB具有出色的可扩展性。除了支持GPT和LLaMA系列模型外，它还允许为自定义模型架构创建工作负载。这种灵活性意味着您可以调整AICB以生成和测试各种模型的通信和计算模式，而不仅仅是预配置选项，使其成为基准测试和优化多样化AI训练框架的宝贵工具。

Custom models can be built using the MockedParam and MockedModel base classes. For specific implementation details, you can refer to the existing MockedMegatron and MockedDeepSpeed implementations.
可以使用MockedParam和MockedModel基类构建自定义模型。有关具体实现细节，您可以参考现有的MockedMegatron和MockedDeepSpeed实现。

Here is an example of a DeepSpeed Llama model:
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

Besides Model, you alse need to mocked the entire training process. The training process for all frameworks is abstracted into the following steps: `init, forward, backward, and step` (excluding pipeline parallelism). We need to further elaborate on the communications that occur in each step with workload items.
除了模型之外，您还需要模拟整个训练过程。所有框架的训练过程被抽象为以下步骤：`init, forward, backward, step`（不包括流水线并行）。我们需要进一步阐述每个步骤中发生的通信以及工作负载项。

In the code, each workload primarily consists of three components:
在代码中，每个工作负载主要由三个组件组成：

1. Communication Information:
   This primarily includes information related to collective communication activities such as `comm_type, comm_group, comm_group_size, and msg_size`.
2. Additional Information:
   This includes supplementary information, such as the source node for broadcast communications and the time for compute operations.
3. Runtime Information:
   This mainly consists of runtime-specific details like `elapsed_time, algo_bw, and bus_bw` which indicate the actual performance and state of the collective communication activities.
Here is a brief example of training process and workload item:
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
    "operation": "init",             # Corresponds to one of init, forward, backward, step
    "comm_type": CommType.broadcast, # The type of communication
    "call_func": "_broadcast_model", # The function invoked in the source code
    "msg_size": param.msg_size(),    # The size of the communication message
    "comm_group": CommGroup.dp_group,# The communication group
    "src": 0,                        # Optional: Only for broadcast, specifies the source node
    "additional": send_next          # Optional: Specifies the corresponding operation in pipeline parallelism
})
```

# Troubleshooting & FAQs
N/A

