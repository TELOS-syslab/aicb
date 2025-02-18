"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from utils.utils import divide, CommType, CommGroup  # 从utils.utils导入divide函数，CommType和CommGroup枚举类型
from workload_generator.mocked_model.MockedModel import MockedModel, Linear, MockedParam  # 从mocked_model导入MockedModel, Linear, MockedParam
from log_analyzer.log import Workload, LogItem  # 从log_analyzer模块导入Workload和LogItem


import os

# import sarathi.metrics.cuda_timer
# import torch

# from vidur.profiling.common.cuda_timer import CudaTimer

# 猴子补丁技术：将 CudaTimer 类替换为 sarathi 实现
# monkey patching the CudaTimer class to use the sarathi implementation
# sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

# from sarathi.model_executor.weight_utils import initialize_dummy_weights

# from workload_generator.profiling.common.model_config import ModelConfig
from vidur.profiling.common.model_config import ModelConfig
# from vidur.profiling.common.timer_stats_store import TimerStatsStore
# from vidur.profiling.mlp.mlp_impl import GPTModel
# from vidur.profiling.utils import ProfileMethod
# from vidur.profiling.utils.record_function_tracer import RecordFunctionTracer

class SarathiModel(MockedModel):

    def __init__(
            self, 
            config,
            model_config: ModelConfig,  # 模型配置
            num_tensor_parallel_workers: int,  # 张量并行的工作者数量
            profile_method: str,  # 轮廓分析方法
            # rank: int,  # 排名
            output_dir: str,  # 输出目录
            ):
        
            pass
    
        # 初始化嵌入层，传入词汇表大小、隐藏层大小、张量并行大小、序列长度和微批次大小
        # self.embedding = MegatronEmbedding(
        #     config.padded_vocab_size,
        #     config.hidden_size,
        #     config.tensor_model_parallel_size,
        #     config.seq_length,
        #     config.micro_batch,
        # )
        # 初始化Transformer层列表，每一层包含隐藏层大小、FFN隐藏层大小、张量并行大小等参数
        # self.layers = [
        #     MegatronTransformorLayer(
        #         config.hidden_size,  # 隐藏层大小
        #         config.ffn_hidden_size,  # FFN隐藏层大小
        #         config.tensor_model_parallel_size,  # 张量并行大小
        #         config.seq_length,  # 序列长度
        #         config.micro_batch,  # 微批次大小
        #         config.num_attention_heads,  # 注意力头数量
        #         i,  # 当前层索引
        #         config.expert_model_parallel_size,  # 专家模型并行大小
        #         config.moe_router_topk,  # MoE路由TopK值
        #         config.num_experts,  # 专家数量
        #         config.moe_grouped_gemm,  # 是否启用MoE分组GEMM
        #         config.enable_sequence_parallel,  # 是否启用序列并行
        #         config.computation_enable,  # 是否启用计算
        #         config.add_bias_linear,  # 是否添加偏置线性
        #         config.moe_enable,  # 是否启用MoE
        #     )
        #     for i in range(config.num_layers)  # 遍历层数，创建每一层
        # ]
        # 初始化最终归一化层，传入隐藏层大小、词汇表大小、张量并行大小等参数
        # self.final_norm = MegatronColumnLinear(
        #     config.hidden_size,  # 隐藏层大小
        #     config.padded_vocab_size,  # 填充后的词汇表大小
        #     config.tensor_model_parallel_size,  # 张量并行大小
        #     config.seq_length,  # 序列长度
        #     config.micro_batch,  # 微批次大小
        #     1,  # 层类型标识为"final"
        #     "final",  # 层名称为"final"
        #     sequence_parallel_enabled=config.enable_sequence_parallel,  # 是否启用序列并行
        #     computation_enable=config.computation_enable,  # 是否启用计算
        #     add_bias_linear=config.add_bias_linear,  # 是否添加偏置线性
        # )

'''
    # /mnt/fth/software4/vidur/vidur/profiling/mlp/mlp_wrapper.py
    def __init__(
        self,
        model_config: ModelConfig,  # 模型配置
        num_tensor_parallel_workers: int,  # 张量并行的工作者数量
        profile_method: str,  # 轮廓分析方法
        rank: int,  # 排名
        output_dir: str,  # 输出目录
    ):
        super().__init__()

        self.timer_stats_store = TimerStatsStore(profile_method=profile_method)  # 计时器统计存储

        self.model_config = model_config  # 模型配置
        self.num_tensor_parallel_workers = num_tensor_parallel_workers  # 张量并行的工作者数量
        self.profile_method = profile_method  # 轮廓分析方法
        self.rank = rank  # 排名
        self.output_dir = output_dir  # 输出目录
        os.makedirs(f"{self.output_dir}/profiler_traces/", exist_ok=True)  # 创建输出目录 

        self.model = GPTModel(
            model_config,
            num_tensor_parallel_workers,
            (
                ACTIVE_STEPS
                if self.profile_method == ProfileMethod.RECORD_FUNCTION.value  # 检查是否使用记录功能方法
                else 1
            ),
        )
        initialize_dummy_weights(self.model)  # 初始化虚拟权重
        self.model = self.model.to(dtype=torch.float16).cuda().eval()  # 设置模型精度为 float16 并在 GPU 上执行



    # /disk1/futianhao/software1/aicb/workload_generator/mocked_model/MockedMegatron.py
    def __init__(self, config):
        # 初始化嵌入层，传入词汇表大小、隐藏层大小、张量并行大小、序列长度和微批次大小
        self.embedding = MegatronEmbedding(
            config.padded_vocab_size,
            config.hidden_size,
            config.tensor_model_parallel_size,
            config.seq_length,
            config.micro_batch,
        )
        # 初始化Transformer层列表，每一层包含隐藏层大小、FFN隐藏层大小、张量并行大小等参数
        self.layers = [
            MegatronTransformorLayer(
                config.hidden_size,  # 隐藏层大小
                config.ffn_hidden_size,  # FFN隐藏层大小
                config.tensor_model_parallel_size,  # 张量并行大小
                config.seq_length,  # 序列长度
                config.micro_batch,  # 微批次大小
                config.num_attention_heads,  # 注意力头数量
                i,  # 当前层索引
                config.expert_model_parallel_size,  # 专家模型并行大小
                config.moe_router_topk,  # MoE路由TopK值
                config.num_experts,  # 专家数量
                config.moe_grouped_gemm,  # 是否启用MoE分组GEMM
                config.enable_sequence_parallel,  # 是否启用序列并行
                config.computation_enable,  # 是否启用计算
                config.add_bias_linear,  # 是否添加偏置线性
                config.moe_enable,  # 是否启用MoE
            )
            for i in range(config.num_layers)  # 遍历层数，创建每一层
        ]
        # 初始化最终归一化层，传入隐藏层大小、词汇表大小、张量并行大小等参数
        self.final_norm = MegatronColumnLinear(
            config.hidden_size,  # 隐藏层大小
            config.padded_vocab_size,  # 填充后的词汇表大小
            config.tensor_model_parallel_size,  # 张量并行大小
            config.seq_length,  # 序列长度
            config.micro_batch,  # 微批次大小
            1,  # 层类型标识为"final"
            "final",  # 层名称为"final"
            sequence_parallel_enabled=config.enable_sequence_parallel,  # 是否启用序列并行
            computation_enable=config.computation_enable,  # 是否启用计算
            add_bias_linear=config.add_bias_linear,  # 是否添加偏置线性
        )

'''