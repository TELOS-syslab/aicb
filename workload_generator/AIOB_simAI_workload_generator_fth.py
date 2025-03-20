"""
Copyright (c) 2021, Alibaba Group;  # 版权声明，归属阿里巴巴集团
Licensed under the Apache License, Version 2.0 (the "License");  # 遵循 Apache 2.0 许可协议
you may not use this file except in compliance with the License.  # 除非符合许可协议，否则不得使用此文件
You may obtain a copy of the License at  # 获取许可协议副本的地址
   http://www.apache.org/licenses/LICENSE-2.0  
Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
distributed under the License is distributed on an "AS IS" BASIS,  # 软件按“现状”分发
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何形式的明示或暗示担保
See the License for the specific language governing permissions and  # 查看许可协议中的具体条款
limitations under the License.  # 许可限制
"""

import workload_generator.mocked_model.MockedDeepspeed  # 导入 MockedDeepspeed 模块
from workload_generator.mocked_model.MockedMegatron import *  # 导入 MockedMegatron 模块的所有内容
from workload_generator.mocked_model.MockedModel import MockedParam, MockedModel  # 导入 MockedParam 和 MockedModel 类
from utils.utils import CommType, get_params, get_comp_out, extract_averages  # 导入工具函数和枚举类型
import os  # 导入操作系统相关模块
from typing import List, Tuple  # 导入类型注解支持
from collections import deque  # 导入双端队列
import dataclasses  # 导入数据类支持
from enum import Enum  # 导入枚举类型支持
import pdb

try:
    import torch  # 尝试导入 PyTorch 库
except ImportError as e:  # 如果导入失败
    torch = None  # 设置 torch 为 None
    print("Failed to import 'torch'.")  # 打印错误信息
import math  # 导入数学库
import re  # 导入正则表达式库


@dataclasses.dataclass  # 定义一个数据类
class Work_Item:
    name: str = dataclasses.field(default="none")  # 工作项名称，默认值为 "none"
    placeholder: int = dataclasses.field(default=-1)  # 占位符，默认值为 -1
    forward_compute_time: int = dataclasses.field(default=0)  # 前向计算时间，默认值为 0
    forward_comm: str = dataclasses.field(default="NONE")  # 前向通信操作类型，默认值为 "NONE"
    forward_comm_size: int = dataclasses.field(default=0)  # 前向通信消息大小，默认值为 0
    backward_compute_time: int = dataclasses.field(default=0)  # 反向计算时间，默认值为 0
    backward_comm: str = dataclasses.field(default="NONE")  # 反向通信操作类型，默认值为 "NONE"
    backward_comm_size: int = dataclasses.field(default=0)  # 反向通信消息大小，默认值为 0
    dp_compute_time: int = dataclasses.field(default=0)  # 数据并行计算时间，默认值为 0
    dp_comm: str = dataclasses.field(default="NONE")  # 数据并行通信操作类型，默认值为 "NONE"
    dp_comm_size: int = dataclasses.field(default=0)  # 数据并行通信消息大小，默认值为 0
    process_time: int = dataclasses.field(default=100)  # 处理时间，默认值为 100


def _get_aiob_compute_time(compute_cache, forward_or_backward, stage):  # 定义获取 AIOB 计算时间的函数
    compute_time_map = compute_cache  # 将计算缓存赋值给 compute_time_map
    if stage == "grad":  # 如果阶段是梯度计算
        prefix = stage + "_" + forward_or_backward  # 构造前缀，例如 "grad_forward"
    elif stage == "embedding":  # 如果阶段是嵌入层
        prefix = "Emb"  # 前缀为 "Emb"
    elif stage == "final":  # 如果阶段是最终层
        prefix = "attention" + "_" + forward_or_backward  # 构造前缀，例如 "attention_forward"
    else:  # 其他情况
        prefix = stage + "_" + forward_or_backward  # 构造前缀，例如 "mlp_forward"

    for key, value in compute_time_map.items():  # 遍历计算缓存中的键值对
        if prefix == key:  # 如果前缀匹配键
            compute_time = compute_time_map.get(key)  # 获取对应的计算时间
            return compute_time  # 返回计算时间
    
    print("[warn] can't match any stage", stage)  # 如果未匹配到任何阶段，打印警告信息
    # pdb.set_trace() # fth
    return 1  # 返回默认值 1

def _get_aiob_compute_time_inference(compute_cache, forward_or_backward, stage):  # 定义获取 AIOB 计算时间的函数
    compute_time_map = compute_cache  # 将计算缓存赋值给 compute_time_map
    if stage == "grad":  # 如果阶段是梯度计算
        prefix = stage + "_" + forward_or_backward  # 构造前缀，例如 "grad_forward"
    elif stage == "embedding":  # 如果阶段是嵌入层
        prefix = "Emb"  # 前缀为 "Emb"
    elif stage == "final":  # 如果阶段是最终层
        prefix = "attention" + "_" + forward_or_backward  # 构造前缀，例如 "attention_forward"
    elif stage == "mlp_layer_prefill_forward":
        prefix == "mlp_layer_prefill_forward"
    elif stage == "attention_layer_prefill_forward":
        prefix == "attention_layer_prefill_forward"
    elif stage == "mlp_layer_decode_forward":
        prefix == "mlp_layer_decode_forward"
    elif stage == "attention_layer_decode_forward":
        prefix == "attention_layer_decode_forward"
    elif stage == "emb":
        prefix == "emb"
    elif stage == "add":
        prefix == "add"
    elif prefix == "mlp":
        prefix == "mlp"
    elif prefix == "att":
        prefix == "att"
    else:  # 其他情况
        prefix = stage + "_" + forward_or_backward  # 构造前缀，例如 "mlp_forward"

    for key, value in compute_time_map.items():  # 遍历计算缓存中的键值对
        if prefix == key:  # 如果前缀匹配键
            compute_time = compute_time_map.get(key)  # 获取对应的计算时间
            return compute_time  # 返回计算时间

    print("[warn] can't match any stage", stage)  # 如果未匹配到任何阶段，打印警告信息
    return 1  # 返回默认值 1


class LayerInfo:  # 定义层信息类
    def __init__(self, layer_id, layer_name, param_count):  # 初始化方法
        self.layer_id = layer_id  # 层 ID
        self.layer_name = layer_name  # 层名称
        self.param_count = param_count  # 参数数量


class SIMAI_workload:  # 定义一个类用于生成工作负载
    def __init__(self, model, args, compute_cache=None):  # 初始化方法
        self.model = model  # 模型对象
        self.args = args  # 参数对象
        self.compute_cache = compute_cache  # 计算时间缓存
        self.workload = []  # 工作负载列表
        self.seq_len = args.seq_length  # 序列长度
        self.tp = args.tensor_model_parallel_size  # 张量并行度
        self.mbs = args.micro_batch  # 微批大小
        if args.moe_enable:  # 如果启用了 MoE（混合专家模型）
            self.expert_model_parallel_size = args.expert_model_parallel_size  # 设置专家模型并行度
            self.num_experts = args.num_experts  # 设置专家数量
            self.topk = args.moe_router_topk  # 设置路由专家数量

    def get_model_details(self):  # 获取模型的详细信息
        layers = []  # 存储层信息的列表
        visited = set()  # 用于记录已访问的模块

        def traverse_model(model):  # 定义递归遍历模型的方法
            if id(model) in visited:  # 如果模块已访问过，跳过
                return
            visited.add(id(model))  # 标记当前模块为已访问

            if self.args.enable_sequence_parallel:  # 如果启用了序列并行
                if (
                    isinstance(model, MegatronColumnLinear)  # 如果是列线性层
                    or isinstance(model, MegatronRowLinear)  # 或行线性层
                    or isinstance(model, MegatronEmbedding)  # 或嵌入层
                    or isinstance(model, FusedLayernorm)  # 或归一化层
                ):
                    params = model.parameters()  # 获取参数
                    param_count = sum(p.numel() for p in params)  # 计算参数总数
                    layers.append(LayerInfo(model.layer_id, model.name, param_count))  # 添加层信息
                if isinstance(model, MOEMLP):  # 如果是 MoE 的 MLP 层
                    moe_params = model.parameters()  # 获取参数
                    moe_param_count = sum(p.numel() for p in moe_params)  # 计算参数总数
                    layers.append(LayerInfo(model.layer_id, model.name, moe_param_count))  # 添加层信息

            else:  # 如果未启用序列并行
                if (
                    isinstance(model, MegatronAttention)  # 如果是注意力层
                    or isinstance(model, MegatronMlp)  # 或 MLP 层
                    or isinstance(model, MegatronEmbedding)  # 或嵌入层
                ):
                    params = model.parameters()  # 获取参数
                    param_count = sum(p.numel() for p in params)  # 计算参数总数
                    layers.append(LayerInfo(model.layer_id, model.name, param_count))  # 添加层信息

            for child in model.child_modules():  # 遍历子模块
                traverse_model(child)  # 递归处理子模块

        traverse_model(model)  # 调用递归函数遍历整个模型

        return layers  # 返回所有层信息

    def _get_total_params(self):  # 获取模型总参数和 MoE 参数
        total_params = 0  # 初始化总参数计数
        moe_param_count = 0  # 初始化 MoE 参数计数
        layers = self.get_model_details()  # 获取模型层信息
        for layer in layers:  # 遍历每一层
            total_params += layer.param_count  # 累加总参数
            if "moe" in layer.layer_name:  # 如果是 MoE 层
                moe_param_count += layer.param_count  # 累加 MoE 参数

        return total_params, moe_param_count  # 返回总参数和 MoE 参数

    def workload_generate_aiob(self):
        # 计算ga_num为全局批次除以（微批次数 × 数据并行数）
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)
        # 如果ga_num小于1，打印警告信息
        if self.ga_num < 1:
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )
        # 设置默认计算时间为1
        default_compute_time = 1
        # 初始化计算时间为0
        compute_time = 0
        # 计算张量并行通信大小
        tp_comm_size = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        )
        # 获取模型的层级详情
        layers = self.get_model_details()
        # 获取模型参数总数和MoE参数数量
        total_params, moe_param_count = self._get_total_params()
        # 注释掉的代码，用于添加一个名为"norm"的工作项
        # self.workload.append(Work_Item(name="norm", forward_compute_time=0,
        #                         forward_comm = "BROADCAST", forward_comm_size= 8*self.args.micro_batch*self.args.seq_length,
        #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
        #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
        #                         ))
        # 获取前向计算时间
        
        forward_compute_time = _get_aiob_compute_time(
            self.compute_cache, "forward", "grad"
        )
        print(f">>fth forward_compute_time:{forward_compute_time} /disk1/futianhao/software1/aicb/workload_generator/AIOB_simAI_workload_generator_fth.py")
        # 获取反向计算时间
        backward_compute_time = _get_aiob_compute_time(
            self.compute_cache, "backward", "grad"
        )
        # 添加名为"grad_gather"的工作项，涉及所有聚集操作
        self.workload.append(
            Work_Item(
                name="grad_gather",
                forward_compute_time=default_compute_time,  # 前向计算时间
                forward_comm="NONE",  # 前向通信方式
                forward_comm_size=0,  # 前向通信大小
                backward_compute_time=default_compute_time,  # 反向计算时间
                backward_comm="NONE",  # 反向通信方式
                backward_comm_size=0,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="ALLGATHER",  # 数据并行通信方式
                dp_comm_size=2 * (total_params - moe_param_count),  # 数据并行通信大小
            )
        )
        # 添加名为"grad_param_comm"的工作项，涉及参数通信
        self.workload.append(
            Work_Item(
                name="grad_param_comm",
                forward_compute_time=default_compute_time,  # 前向计算时间
                forward_comm="NONE",  # 前向通信方式
                forward_comm_size=0,  # 前向通信大小
                backward_compute_time=default_compute_time,  # 反向计算时间
                backward_comm="NONE",  # 反向通信方式
                backward_comm_size=0,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="REDUCESCATTER",  # 数据并行通信方式
                dp_comm_size=4 * (total_params - moe_param_count),  # 数据并行通信大小
            )
        )
        # 添加名为"grad_param_compute"的工作项，涉及参数计算
        self.workload.append(
            Work_Item(
                name="grad_param_compute",
                forward_compute_time=default_compute_time,  # 前向计算时间
                forward_comm="NONE",  # 前向通信方式
                forward_comm_size=0,  # 前向通信大小
                backward_compute_time=forward_compute_time + backward_compute_time,  # 反向计算时间
                backward_comm="NONE",  # 反向通信方式
                backward_comm_size=0,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="NONE",  # 数据并行通信方式
                dp_comm_size=0,  # 数据并行通信大小
            )
        )
        # 如果不启用序列并行，添加"layernorm"的工作项
        if not self.args.enable_sequence_parallel:
            self.workload.append(
                Work_Item(
                    name="layernorm",
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm="NONE",  # 前向通信方式
                    forward_comm_size=0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="ALLREDUCE",  # 反向通信方式
                    backward_comm_size=2 * total_params,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )
        # 根据张量模型并行大小设置嵌入层反向通信方式
        if args.tensor_model_parallel_size == 1 :
            emd_backward_comm = "NONE"  # 如果并行度为1，无需通信
        else:
            emd_backward_comm = "ALLREDUCE"  # 否则，使用ALLREDUCE通信
        # 添加名为"embedding_grads"的工作项，涉及嵌入梯度的通信
        self.workload.append(
            Work_Item(
                name="embedding_grads",
                forward_compute_time=default_compute_time,  # 前向计算时间
                forward_comm="NONE",  # 前向通信方式
                forward_comm_size=0,  # 前向通信大小
                backward_compute_time=default_compute_time,  # 反向计算时间
                backward_comm=emd_backward_comm,  # 反向通信方式
                backward_comm_size=tp_comm_size,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="NONE",  # 数据并行通信方式
                dp_comm_size=0,  # 数据并行通信大小
            )
        )
        # 如果专家模型并行数不等于数据并行数，添加MoE梯度规范化的工作项
        if self.args.expert_model_parallel_size != self.args.dp_num:
            self.workload.append(Work_Item(
                    name="moe_grad_norm1", 
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm = "NONE",  # 前向通信方式
                    forward_comm_size= 0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="ALLGATHER_DP_EP",  # 数据并行通信方式
                    dp_comm_size=2*moe_param_count  # 数据并行通信大小
                ))
            self.workload.append(Work_Item(
                    name="moe_grad_norm2", 
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm = "NONE",  # 前向通信方式
                    forward_comm_size= 0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="REDUCESCATTER_DP_EP",  # 数据并行通信方式
                    dp_comm_size=4*moe_param_count  # 数据并行通信大小
                ))
        # 根据ga_num的值循环添加工作项
        for _ in range(self.ga_num):
            for layer in layers:
                name = layer.layer_name  # 获取当前层的名称
                forward_comm = backward_comm = backward_comm_2 = "NONE"  # 初始化通信方式
                forward_comm_size = tp_comm_size  # 设置前向通信大小
                emb_comm_size = tp_comm_size  # 设置嵌入通信大小
                backward_comm_size = 0  # 初始化反向通信大小
                dp_comm = "NONE"  # 初始化数据并行通信方式
                dp_comm_size = 0  # 初始化数据并行通信大小
                # 如果启用序列并行
                if self.args.enable_sequence_parallel:
                    # 如果当前层名包含"embedding"
                    if "embedding" in name:
                        # 根据张量模型并行大小设置通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"  # 无前向通信
                            backward_comm = "NONE"  # 无反向通信
                        else:
                            forward_comm = "ALLREDUCE"  # 使用ALLREDUCE进行前向通信
                            backward_comm = "NONE"  # 无反向通信
                        # 获取嵌入层的计算时间
                        emb_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "", "embedding"
                        )
                        # 添加当前嵌入层的工作项
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=emb_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
                    # 如果当前层名包含"row"
                    if "row" in name:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )

                        # 如果启用了激活重计算且层名包含'attention'，则前向计算时间加倍
                        if self.args.recompute_activations and 'attention' in name:
                            forward_compute_time *= 2
                        # 将前向和反向计算时间减半
                        forward_compute_time = int(forward_compute_time / 2)
                        backward_compute_time = int(backward_compute_time / 2)
                        # 设置序列并行的前向通信大小
                        forward_comm_size_sp = tp_comm_size
                        # 根据张量模型并行大小设置通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"  # 无前向通信
                            backward_comm = "NONE"  # 无反向通信
                        else:
                            forward_comm = "REDUCESCATTER"  # 使用REDUCESCATTER进行前向通信
                            backward_comm = "ALLGATHER"  # 使用ALLGATHER进行反向通信
                        # 添加当前"row"层的工作项
                        self.workload.append(
                                Work_Item(
                                    name=name,
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm=forward_comm,  # 前向通信方式
                                    forward_comm_size=forward_comm_size,  # 前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=backward_comm,  # 反向通信方式
                                    backward_comm_size=forward_comm_size_sp,  # 反向通信大小（序列并行重叠的ALLGATHER）
                                    dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size,  # 数据并行通信大小
                                )
                            )

                    # 如果当前层名包含"column"
                    elif "column" in name:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )

                        # 如果启用了激活重计算且层名包含'attention'，则前向计算时间加倍
                        if self.args.recompute_activations and 'attention' in name:
                            forward_compute_time *= 2
                        # 将前向和反向计算时间减半
                        forward_compute_time = int(forward_compute_time / 2)
                        backward_compute_time = int(backward_compute_time / 2)
                        # 根据张量模型并行大小设置通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"  # 无前向通信
                            backward_comm = "NONE"  # 无反向通信
                            backward_comm_2 = "NONE"  # 无第二次反向通信
                        else:
                            forward_comm = "ALLGATHER"  # 使用ALLGATHER进行前向通信
                            backward_comm = "REDUCESCATTER"  # 使用REDUCESCATTER进行第一次反向通信
                            backward_comm_2 = "ALLGATHER"  # 使用ALLGATHER进行第二次反向通信
                        # 添加当前"column"层的工作项
                        self.workload.append(
                                Work_Item(
                                    name=name,
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm=forward_comm,  # 前向通信方式
                                    forward_comm_size=forward_comm_size,  # 前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=backward_comm,  # 反向通信方式
                                    backward_comm_size=backward_comm_size,  # 反向通信大小
                                    dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size,  # 数据并行通信大小
                                )
                            )
                    # 如果当前层名包含"moelayer"
                    elif "moelayer" in name:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        # 根据张量模型并行大小设置多个通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm1 = "NONE"  # 第一阶段前向通信方式
                            forward_comm2 = "NONE"  # 第二阶段前向通信方式
                            forward_comm3 = "ALLTOALL_EP"  # 第三阶段前向通信方式
                            forward_comm4 = "NONE"  # 第四阶段前向通信方式
                            forward_comm5 = "NONE"  # 第五阶段前向通信方式
                            forward_comm6 = "ALLTOALL_EP"  # 第六阶段前向通信方式
                            forward_comm7 = "NONE"  # 第七阶段前向通信方式
                        else:
                            forward_comm1 = "ALLGATHER"  # 第一阶段前向通信方式
                            forward_comm2 = "ALLTOALL"  # 第二阶段前向通信方式
                            forward_comm3 = "ALLTOALL_EP"  # 第三阶段前向通信方式
                            forward_comm4 = "ALLGATHER"  # 第四阶段前向通信方式
                            forward_comm5 = "REDUCESCATTER"  # 第五阶段前向通信方式
                            forward_comm6 = "ALLTOALL_EP"  # 第六阶段前向通信方式
                            forward_comm7 = "ALLTOALL"  # 第七阶段前向通信方式
                        # 如果专家模型并行数不为1，添加多个工作项
                        if args.expert_model_parallel_size != 1:
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm1,  # 第一阶段前向通信方式
                                    forward_comm_size= 2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm1,  # 第一阶段反向通信方式
                                    backward_comm_size=2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm2,  # 第二阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第二阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm2,  # 第二阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第二阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm3,  # 第三阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第三阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm3,  # 第三阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第三阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm4,  # 第四阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第四阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm5,  # 第五阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第五阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm5,  # 第五阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第五阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm4,  # 第四阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm6,  # 第六阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第六阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm6,  # 第六阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第六阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm7,  # 第七阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第七阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm7,  # 第七阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第七阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                        else:
                            # 如果专家模型并行数为1，添加较少的工作项
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm1,  # 第一阶段前向通信方式
                                    forward_comm_size= 2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm1,  # 第一阶段反向通信方式
                                    backward_comm_size=2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm2,  # 第二阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第二阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm2,  # 第二阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第二阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm3,  # 第三阶段前向通信方式
                                    forward_comm_size=1,  # 第三阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm3,  # 第三阶段反向通信方式
                                    backward_comm_size=1,  # 第三阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm4,  # 第四阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第四阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm4,  # 第四阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm5,  # 第五阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第五阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm4,  # 第四阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm6,  # 第六阶段前向通信方式
                                    forward_comm_size=1,  # 第六阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm6,  # 第六阶段反向通信方式
                                    backward_comm_size=1,  # 第六阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm7,  # 第七阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第七阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm7,  # 第七阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第七阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                else:
                    # 如果不启用序列并行，根据张量模型并行大小设置通信方式
                    if args.tensor_model_parallel_size == 1 :
                        forward_comm = "NONE"  # 无前向通信
                        backward_comm = "NONE"  # 无反向通信
                    else:
                        forward_comm = "ALLREDUCE"  # 使用ALLREDUCE进行前向通信
                        backward_comm = "NONE"  # 无反向通信
                    # 如果启用了激活重计算且层名包含'attention'，则前向计算时间加倍
                    if self.args.recompute_activations and 'attention' in name:
                        forward_compute_time *= 2
                    # 如果当前层名包含"embedding"
                    if "embedding" in name:
                        # 获取嵌入层的计算时间
                        emb_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "", "embedding"
                        )
                        # 添加当前嵌入层的工作项
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
                    else:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        # 添加当前层的工作项
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=forward_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=backward_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
            # 注释掉的代码，用于添加嵌入层归一化的工作项
            # compute_time = _get_aiob_compute_time(self.compute_cache, "forward", "embedding")
            # self.workload.append(Work_Item(name="embedding_norm", forward_compute_time=compute_time,
            #                         forward_comm = "ALLREDUCE", forward_comm_size= self.args.vocab_size*self.args.hidden_size*2,
            #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
            #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
            #                         ))
        # 添加三个"cross_entropy"的工作项
        for i in range(3):
            self.workload.append(
                Work_Item(
                    name="cross_entropy" + str(i + 1),  # 工作项名称
                    forward_compute_time=compute_time,  # 前向计算时间
                    forward_comm="ALLREDUCE",  # 前向通信方式
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,  # 前向通信大小
                    backward_compute_time=compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )

        # 添加四个"optimizer"的工作项
        for i in range(4):
            self.workload.append(
                Work_Item(
                    name="optimizer" + str(i + 1),  # 工作项名称
                    forward_compute_time=compute_time,  # 前向计算时间
                    forward_comm="ALLREDUCE",  # 前向通信方式
                    forward_comm_size=4,  # 前向通信大小
                    backward_compute_time=compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )

    def workload_generate_aiob_inference(self):
        # print(f'?? fth workload_generate_aiob_inference')
        # 计算ga_num为全局批次除以（微批次数 × 数据并行数）
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)
        # 如果ga_num小于1，打印警告信息
        if self.ga_num < 1:
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"
            )
        # 设置默认计算时间为1
        default_compute_time = 1
        # 初始化计算时间为0
        compute_time = 0
        # 计算张量并行通信大小
        tp_comm_size = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size
        )
        # 获取模型的层级详情
        layers = self.get_model_details()
        # 获取模型参数总数和MoE参数数量
        total_params, moe_param_count = self._get_total_params()
        # 注释掉的代码，用于添加一个名为"norm"的工作项
        # self.workload.append(Work_Item(name="norm", forward_compute_time=0,
        #                         forward_comm = "BROADCAST", forward_comm_size= 8*self.args.micro_batch*self.args.seq_length,
        #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
        #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
        #                         ))
        # 获取前向计算时间
        
        if args.is_inference == False:
            forward_compute_time = _get_aiob_compute_time(
                self.compute_cache, "forward", "grad"
            )
            print(f">>fth forward_compute_time:{forward_compute_time} /disk1/futianhao/software1/aicb/workload_generator/AIOB_simAI_workload_generator_fth.py")
        # 获取反向计算时间 inference等于0等于0 不用管
        if args.is_inference == False:
            backward_compute_time = _get_aiob_compute_time(
                self.compute_cache, "backward", "grad"
            )
        # print(f"??fth backward_compute_time={backward_compute_time}")
        # 添加名为"grad_gather"的工作项，涉及所有聚集操作
        if args.is_inference == False:
            self.workload.append(
                Work_Item(
                    name="grad_gather",
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm="NONE",  # 前向通信方式
                    forward_comm_size=0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="ALLGATHER",  # 数据并行通信方式
                    dp_comm_size=2 * (total_params - moe_param_count),  # 数据并行通信大小
                )
            )
            # 添加名为"grad_param_comm"的工作项，涉及参数通信
            self.workload.append(
                Work_Item(
                    name="grad_param_comm",
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm="NONE",  # 前向通信方式
                    forward_comm_size=0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="REDUCESCATTER",  # 数据并行通信方式
                    dp_comm_size=4 * (total_params - moe_param_count),  # 数据并行通信大小
                )
            )

            # 添加名为"grad_param_compute"的工作项，涉及参数计算
            self.workload.append(
                Work_Item(
                    name="grad_param_compute",
                    # forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm="NONE",  # 前向通信方式
                    forward_comm_size=0,  # 前向通信大小
                    backward_compute_time=forward_compute_time + backward_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )

        if args.is_inference == False:

            # 如果不启用序列并行，添加"layernorm"的工作项
            if not self.args.enable_sequence_parallel:
                self.workload.append(
                    Work_Item(
                        name="layernorm",
                        forward_compute_time=default_compute_time,  # 前向计算时间
                        forward_comm="NONE",  # 前向通信方式
                        forward_comm_size=0,  # 前向通信大小
                        backward_compute_time=default_compute_time,  # 反向计算时间
                        backward_comm="ALLREDUCE",  # 反向通信方式
                        backward_comm_size=2 * total_params,  # 反向通信大小
                        dp_compute_time=default_compute_time,  # 数据并行计算时间
                        dp_comm="NONE",  # 数据并行通信方式
                        dp_comm_size=0,  # 数据并行通信大小
                    )
                )

        elif args.is_inference == True:
            # 如果不启用序列并行，添加"layernorm"的工作项
            if not self.args.enable_sequence_parallel:
                self.workload.append(
                    Work_Item(
                        name="layernorm",
                        forward_compute_time=default_compute_time,  # 前向计算时间
                        forward_comm="NONE",  # 前向通信方式
                        forward_comm_size=0,  # 前向通信大小
                        backward_compute_time=default_compute_time,  # 反向计算时间
                        backward_comm="NONE",  # 反向通信方式
                        backward_comm_size=0,  # 反向通信大小
                        dp_compute_time=default_compute_time,  # 数据并行计算时间
                        dp_comm="NONE",  # 数据并行通信方式
                        dp_comm_size=0,  # 数据并行通信大小
                    )
                )

        # 根据张量模型并行大小设置嵌入层反向通信方式
        if args.tensor_model_parallel_size == 1 :
            emd_backward_comm = "NONE"  # 如果并行度为1，无需通信
        else:
            emd_backward_comm = "ALLREDUCE"  # 否则，使用ALLREDUCE通信
        # 添加名为"embedding_grads"的工作项，涉及嵌入梯度的通信
        self.workload.append(
            Work_Item(
                name="embedding_grads",
                forward_compute_time=default_compute_time,  # 前向计算时间
                forward_comm="NONE",  # 前向通信方式
                forward_comm_size=0,  # 前向通信大小
                backward_compute_time=default_compute_time,  # 反向计算时间
                backward_comm=emd_backward_comm,  # 反向通信方式
                backward_comm_size=tp_comm_size,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="NONE",  # 数据并行通信方式
                dp_comm_size=0,  # 数据并行通信大小
            )
        )
        # 如果专家模型并行数不等于数据并行数，添加MoE梯度规范化的工作项
        if self.args.expert_model_parallel_size != self.args.dp_num:
            self.workload.append(Work_Item(
                    name="moe_grad_norm1", 
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm = "NONE",  # 前向通信方式
                    forward_comm_size= 0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="ALLGATHER_DP_EP",  # 数据并行通信方式
                    dp_comm_size=2*moe_param_count  # 数据并行通信大小
                ))
            self.workload.append(Work_Item(
                    name="moe_grad_norm2", 
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm = "NONE",  # 前向通信方式
                    forward_comm_size= 0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="REDUCESCATTER_DP_EP",  # 数据并行通信方式
                    dp_comm_size=4*moe_param_count  # 数据并行通信大小
                ))
        # 根据ga_num的值循环添加工作项
        for _ in range(self.ga_num):
            for layer in layers:
                name = layer.layer_name  # 获取当前层的名称
                forward_comm = backward_comm = backward_comm_2 = "NONE"  # 初始化通信方式
                forward_comm_size = tp_comm_size  # 设置前向通信大小
                emb_comm_size = tp_comm_size  # 设置嵌入通信大小
                backward_comm_size = 0  # 初始化反向通信大小
                dp_comm = "NONE"  # 初始化数据并行通信方式
                dp_comm_size = 0  # 初始化数据并行通信大小
                # 如果启用序列并行
                if self.args.enable_sequence_parallel:
                    # 如果当前层名包含"embedding"
                    if "embedding" in name:
                        # 根据张量模型并行大小设置通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"  # 无前向通信
                            backward_comm = "NONE"  # 无反向通信
                        else:
                            forward_comm = "ALLREDUCE"  # 使用ALLREDUCE进行前向通信
                            backward_comm = "NONE"  # 无反向通信
                        # 获取嵌入层的计算时间
                        emb_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "", "embedding"
                        )
                        # 添加当前嵌入层的工作项
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=emb_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
                    # 如果当前层名包含"row"
                    if "row" in name:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )

                        # 如果启用了激活重计算且层名包含'attention'，则前向计算时间加倍
                        if self.args.recompute_activations and 'attention' in name:
                            forward_compute_time *= 2
                        # 将前向和反向计算时间减半
                        forward_compute_time = int(forward_compute_time / 2)
                        backward_compute_time = int(backward_compute_time / 2)
                        # 设置序列并行的前向通信大小
                        forward_comm_size_sp = tp_comm_size
                        # 根据张量模型并行大小设置通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"  # 无前向通信
                            backward_comm = "NONE"  # 无反向通信
                        else:
                            forward_comm = "REDUCESCATTER"  # 使用REDUCESCATTER进行前向通信
                            backward_comm = "ALLGATHER"  # 使用ALLGATHER进行反向通信
                        # 添加当前"row"层的工作项
                        self.workload.append(
                                Work_Item(
                                    name=name,
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm=forward_comm,  # 前向通信方式
                                    forward_comm_size=forward_comm_size,  # 前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=backward_comm,  # 反向通信方式
                                    backward_comm_size=forward_comm_size_sp,  # 反向通信大小（序列并行重叠的ALLGATHER）
                                    dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size,  # 数据并行通信大小
                                )
                            )

                    # 如果当前层名包含"column"
                    elif "column" in name:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )

                        # 如果启用了激活重计算且层名包含'attention'，则前向计算时间加倍
                        if self.args.recompute_activations and 'attention' in name:
                            forward_compute_time *= 2
                        # 将前向和反向计算时间减半
                        forward_compute_time = int(forward_compute_time / 2)
                        backward_compute_time = int(backward_compute_time / 2)
                        # 根据张量模型并行大小设置通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm = "NONE"  # 无前向通信
                            backward_comm = "NONE"  # 无反向通信
                            backward_comm_2 = "NONE"  # 无第二次反向通信
                        else:
                            forward_comm = "ALLGATHER"  # 使用ALLGATHER进行前向通信
                            backward_comm = "REDUCESCATTER"  # 使用REDUCESCATTER进行第一次反向通信
                            backward_comm_2 = "ALLGATHER"  # 使用ALLGATHER进行第二次反向通信
                        # 添加当前"column"层的工作项
                        self.workload.append(
                                Work_Item(
                                    name=name,
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm=forward_comm,  # 前向通信方式
                                    forward_comm_size=forward_comm_size,  # 前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=backward_comm,  # 反向通信方式
                                    backward_comm_size=backward_comm_size,  # 反向通信大小
                                    dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size,  # 数据并行通信大小
                                )
                            )
                    # 如果当前层名包含"moelayer"
                    elif "moelayer" in name:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        # 根据张量模型并行大小设置多个通信方式
                        if args.tensor_model_parallel_size == 1 :
                            forward_comm1 = "NONE"  # 第一阶段前向通信方式
                            forward_comm2 = "NONE"  # 第二阶段前向通信方式
                            forward_comm3 = "ALLTOALL_EP"  # 第三阶段前向通信方式
                            forward_comm4 = "NONE"  # 第四阶段前向通信方式
                            forward_comm5 = "NONE"  # 第五阶段前向通信方式
                            forward_comm6 = "ALLTOALL_EP"  # 第六阶段前向通信方式
                            forward_comm7 = "NONE"  # 第七阶段前向通信方式
                        else:
                            forward_comm1 = "ALLGATHER"  # 第一阶段前向通信方式
                            forward_comm2 = "ALLTOALL"  # 第二阶段前向通信方式
                            forward_comm3 = "ALLTOALL_EP"  # 第三阶段前向通信方式
                            forward_comm4 = "ALLGATHER"  # 第四阶段前向通信方式
                            forward_comm5 = "REDUCESCATTER"  # 第五阶段前向通信方式
                            forward_comm6 = "ALLTOALL_EP"  # 第六阶段前向通信方式
                            forward_comm7 = "ALLTOALL"  # 第七阶段前向通信方式
                        # 如果专家模型并行数不为1，添加多个工作项
                        if args.expert_model_parallel_size != 1:
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm1,  # 第一阶段前向通信方式
                                    forward_comm_size= 2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm1,  # 第一阶段反向通信方式
                                    backward_comm_size=2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm2,  # 第二阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第二阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm2,  # 第二阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第二阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm3,  # 第三阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第三阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm3,  # 第三阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第三阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm4,  # 第四阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第四阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm5,  # 第五阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第五阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm5,  # 第五阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第五阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm4,  # 第四阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm6,  # 第六阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第六阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm6,  # 第六阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第六阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm7,  # 第七阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第七阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm7,  # 第七阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第七阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                        else:
                            # 如果专家模型并行数为1，添加较少的工作项
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=forward_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm1,  # 第一阶段前向通信方式
                                    forward_comm_size= 2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段前向通信大小
                                    backward_compute_time=backward_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm1,  # 第一阶段反向通信方式
                                    backward_comm_size=2*self.mbs*self.seq_len*self.num_experts,  # 第一阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm2,  # 第二阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第二阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm2,  # 第二阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第二阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm3,  # 第三阶段前向通信方式
                                    forward_comm_size=1,  # 第三阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm3,  # 第三阶段反向通信方式
                                    backward_comm_size=1,  # 第三阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm4,  # 第四阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第四阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm4,  # 第四阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm5,  # 第五阶段前向通信方式
                                    forward_comm_size= tp_comm_size*self.topk,  # 第五阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm4,  # 第四阶段反向通信方式
                                    backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm6,  # 第六阶段前向通信方式
                                    forward_comm_size=1,  # 第六阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm6,  # 第六阶段反向通信方式
                                    backward_comm_size=1,  # 第六阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                            self.workload.append(Work_Item(
                                    name=name, 
                                    forward_compute_time=default_compute_time,  # 前向计算时间
                                    forward_comm = forward_comm7,  # 第七阶段前向通信方式
                                    forward_comm_size= tp_comm_size//self.tp,  # 第七阶段前向通信大小
                                    backward_compute_time=default_compute_time,  # 反向计算时间
                                    backward_comm=forward_comm7,  # 第七阶段反向通信方式
                                    backward_comm_size=tp_comm_size//self.tp,  # 第七阶段反向通信大小
                                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                                    dp_comm=dp_comm,  # 数据并行通信方式
                                    dp_comm_size=dp_comm_size  # 数据并行通信大小
                                    ))
                else:
                    # 如果不启用序列并行，根据张量模型并行大小设置通信方式
                    if args.tensor_model_parallel_size == 1 :
                        forward_comm = "NONE"  # 无前向通信
                        backward_comm = "NONE"  # 无反向通信
                    else:
                        forward_comm = "ALLREDUCE"  # 使用ALLREDUCE进行前向通信
                        backward_comm = "NONE"  # 无反向通信
                    # 如果启用了激活重计算且层名包含'attention'，则前向计算时间加倍
                    if self.args.recompute_activations and 'attention' in name:
                        forward_compute_time *= 2
                    # 如果当前层名包含"embedding"
                    if "embedding" in name:
                        # 获取嵌入层的计算时间
                        emb_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "", "embedding"
                        )
                        # 添加当前嵌入层的工作项
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=emb_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
                    else:
                        # 获取当前层前向和反向的计算时间
                        forward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "forward", name.split("_")[0]
                        )
                        backward_compute_time = _get_aiob_compute_time(
                            self.compute_cache, "backward", name.split("_")[0]
                        )
                        # 添加当前层的工作项
                        self.workload.append(
                            Work_Item(
                                name=name,
                                forward_compute_time=forward_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=backward_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )
            # 注释掉的代码，用于添加嵌入层归一化的工作项
            # compute_time = _get_aiob_compute_time(self.compute_cache, "forward", "embedding")
            # self.workload.append(Work_Item(name="embedding_norm", forward_compute_time=compute_time,
            #                         forward_comm = "ALLREDUCE", forward_comm_size= self.args.vocab_size*self.args.hidden_size*2,
            #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
            #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
            #                         ))
        # 添加三个"cross_entropy"的工作项
        for i in range(3):
            self.workload.append(
                Work_Item(
                    name="cross_entropy" + str(i + 1),  # 工作项名称
                    forward_compute_time=compute_time,  # 前向计算时间
                    forward_comm="ALLREDUCE",  # 前向通信方式
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,  # 前向通信大小
                    backward_compute_time=compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )

        # 添加四个"optimizer"的工作项
        for i in range(4):
            self.workload.append(
                Work_Item(
                    name="optimizer" + str(i + 1),  # 工作项名称
                    forward_compute_time=compute_time,  # 前向计算时间
                    forward_comm="ALLREDUCE",  # 前向通信方式
                    forward_comm_size=4,  # 前向通信大小
                    backward_compute_time=compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )


    def workload_generate(self):
        # 计算ga_num为全局批次除以（微批次数 × 数据并行数）
        self.ga_num = self.args.global_batch // (self.args.micro_batch * self.args.dp_num)  # 计算全局批次数
        # 如果ga_num小于1，打印警告信息
        if self.ga_num < 1:  # 检查ga_num是否小于1
            print(
                "[WARN]: ga num < 1, please confirm global_batch num and micro_batch num"  # 输出警告信息
            )
        # 设置默认计算时间为1
        default_compute_time = 1  # 初始化默认计算时间
        # 初始化计算时间为0
        compute_time = 0  # 初始化计算时间
        # 计算张量并行通信大小
        tp_comm_size = (
            2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size  # 计算张量并行通信大小
        )
        # 获取模型的层级详情
        layers = self.get_model_details()  # 获取模型层详情
        # 获取模型参数总数和MoE参数数量
        total_params, moe_param_count = self._get_total_params()  # 获取总参数和MoE参数数量
        # 注释掉的打印总参数信息
        # print(f"Total params is {total_params}, moe params is {moe_param_count}")  # 打印参数信息
        # 注释掉的添加一个名为"norm"的工作项
        # self.workload.append(Work_Item(name="norm", forward_compute_time=0,
        #                         forward_comm = "BROADCAST", forward_comm_size= 8*self.args.micro_batch*self.args.seq_length,
        #                         backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,
        #                         dp_compute_time=default_compute_time, dp_comm="NONE", dp_comm_size=0
        #                         ))  # 添加归一化工作项
        # 设置前向计算时间为默认值
        forward_compute_time = default_compute_time  # 前向计算时间
        # 设置反向计算时间为默认值
        backward_compute_time = default_compute_time  # 反向计算时间
        # 添加名为"grad_norm"的工作项，涉及梯度归一化和通信
        self.workload.append(
            Work_Item(
                name="grad_norm",  # 工作项名称
                forward_compute_time=forward_compute_time,  # 前向计算时间
                forward_comm="ALLGATHER",  # 前向通信方式
                forward_comm_size=2 * total_params,  # 前向通信大小
                backward_compute_time=backward_compute_time,  # 反向计算时间
                backward_comm="NONE",  # 反向通信方式
                backward_comm_size=0,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="REDUCESCATTER",  # 数据并行通信方式
                dp_comm_size=4 * total_params,  # 数据并行通信大小
            )
        )
        # 如果不启用序列并行，添加"layernorm"的工作项
        if not self.args.enable_sequence_parallel:  # 检查是否启用序列并行
            self.workload.append(
                Work_Item(
                    name="layernorm",  # 工作项名称
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm="NONE",  # 前向通信方式
                    forward_comm_size=0,  # 前向通信大小
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="ALLREDUCE",  # 反向通信方式
                    backward_comm_size=2 * total_params,  # 反向通信大小
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )
        # 如果专家模型并行数不等于数据并行数，添加MoE梯度规范化的工作项
        if args.expert_model_parallel_size != args.dp_num:  # 检查专家模型并行数是否等于数据并行数
            self.workload.append(Work_Item(name="moe_grad_norm1", forward_compute_time=default_compute_time,  # 工作项名称及计算时间
                                    forward_comm = "NONE", forward_comm_size= 0,  # 前向通信方式及大小
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,  # 反向计算及通信
                                    dp_compute_time=default_compute_time, dp_comm="ALLGATHER_DP_EP", dp_comm_size=2*moe_param_count  # 数据并行计算及通信
                                    ))  # 添加MoE梯度规范化1
            self.workload.append(Work_Item(name="moe_grad_norm2", forward_compute_time=default_compute_time,  # 工作项名称及计算时间
                                    forward_comm = "NONE", forward_comm_size= 0,  # 前向通信方式及大小
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,  # 反向计算及通信
                                    dp_compute_time=default_compute_time, dp_comm="REDUCESCATTER_DP_EP", dp_comm_size=4*moe_param_count  # 数据并行计算及通信
                                    ))  # 添加MoE梯度规范化2
        # 根据ga_num的值循环添加工作项
        for _ in range(self.ga_num):  # 循环ga_num次
            for layer in layers:  # 遍历每一层
                name = layer.layer_name  # 获取当前层的名称
                forward_comm = backward_comm = backward_comm_2 = "NONE"  # 初始化通信方式为"NONE"
                forward_comm_size = tp_comm_size  # 设置前向通信大小
                backward_comm_size = tp_comm_size  # 设置反向通信大小
                dp_comm = "NONE"  # 数据并行通信方式
                dp_comm_size = 0  # 数据并行通信大小
                # 如果启用序列并行
                if self.args.enable_sequence_parallel:  # 检查是否启用序列并行
                    # 如果当前层名包含"embedding"
                    if "embedding" in name:  # 检查层名是否包含"embedding"
                        self.workload.append(
                            Work_Item(
                                name=name,  # 工作项名称
                                forward_compute_time=default_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )  # 添加嵌入层的工作项

                    # 如果当前层名包含"row"
                    if "row" in name:  # 检查层名是否包含"row"
                        if self.args.recompute_activations and 'attention' in name:  # 检查是否需要重新计算激活且层名包含"attention"
                            forward_comm_size *= 2  # 将前向通信大小翻倍
                        forward_comm = "REDUCESCATTER"  # 设置前向通信方式为"REDUCESCATTER"
                        backward_comm = "ALLGATHER"  # 设置反向通信方式为"ALLGATHER"
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 添加"row"层的工作项
                                    forward_comm = forward_comm, forward_comm_size= forward_comm_size,  # 前向通信设置
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=tp_comm_size,  # 反向通信设置
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"row"层的工作项
                    # 如果当前层名包含"column"
                    if "column" in name:  # 检查层名是否包含"column"
                        if self.args.recompute_activations and 'attention' in name:  # 检查是否需要重新计算激活且层名包含"attention"
                            forward_comm_size *= 2  # 将前向通信大小翻倍
                        forward_comm = "ALLGATHER"  # 设置前向通信方式为"ALLGATHER"
                        forward_comm2 = "NONE"  # 设置第二阶段前向通信方式为"NONE"
                        backward_comm = "REDUCESCATTER"  # 设置反向通信方式为"REDUCESCATTER"
                        backward_comm_2 = "ALLGATHER"  # 设置第二阶段反向通信方式为"ALLGATHER"
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 添加"column"层的工作项
                                    forward_comm = forward_comm, forward_comm_size= forward_comm_size,  # 前向通信设置
                                    backward_compute_time=default_compute_time, backward_comm="NONE", backward_comm_size=0,  # 反向通信设置
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"column"层的工作项
                    # 如果当前层名包含"moelayer"
                    if "moelayer" in name:  # 检查层名是否包含"moelayer"
                        forward_comm1 = "ALLGATHER"  # 第一阶段前向通信方式
                        forward_comm2 = "ALLTOALL"  # 第二阶段前向通信方式
                        forward_comm3 = "ALLTOALL_EP"  # 第三阶段前向通信方式
                        forward_comm4 = "ALLGATHER"  # 第四阶段前向通信方式
                        forward_comm5 = "REDUCESCATTER"  # 第五阶段前向通信方式
                        forward_comm6 = "ALLTOALL_EP"  # 第六阶段前向通信方式
                        forward_comm7 = "ALLTOALL"  # 第七阶段前向通信方式
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 添加"moelayer"层的多个工作项
                                    forward_comm = forward_comm1, forward_comm_size= 2*self.seq_len*self.num_experts,  # 第一阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm1, backward_comm_size=2*self.seq_len*self.num_experts,  # 第一阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第一个工作项
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 第二个工作项
                                    forward_comm = forward_comm2, forward_comm_size= tp_comm_size//self.tp,  # 第二阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm2, backward_comm_size=tp_comm_size//self.tp,  # 第二阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第二个工作项
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 第三个工作项
                                    forward_comm = forward_comm3, forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第三阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm3, backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第三阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第三个工作项
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 第四个工作项
                                    forward_comm = forward_comm4, forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第四阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm4, backward_comm_size=tp_comm_size*self.topk,  # 第四阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第四个工作项
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 第五个工作项
                                    forward_comm = forward_comm5, forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第五阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm4, backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第五阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第五个工作项
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 第六个工作项
                                    forward_comm = forward_comm6, forward_comm_size= tp_comm_size*self.topk//self.tp,  # 第六阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm6, backward_comm_size=tp_comm_size*self.topk//self.tp,  # 第六阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第六个工作项
                        self.workload.append(Work_Item(name=name, forward_compute_time=default_compute_time,  # 第七个工作项
                                    forward_comm = forward_comm7, forward_comm_size= tp_comm_size//self.tp,  # 第七阶段前向通信
                                    backward_compute_time=default_compute_time, backward_comm=forward_comm7, backward_comm_size=tp_comm_size//self.tp,  # 第七阶段反向通信
                                    dp_compute_time=default_compute_time, dp_comm=dp_comm, dp_comm_size=dp_comm_size  # 数据并行设置
                                    ))  # 添加"moelayer"层的第七个工作项
                    
                else:  # 如果不启用序列并行
                    forward_comm = "ALLREDUCE"  # 设置前向通信方式为"ALLREDUCE"
                    backward_comm = "ALLREDUCE"  # 设置反向通信方式为"ALLREDUCE"
                    if self.args.recompute_activations and 'attention' in name:  # 检查是否需要重新计算激活且层名包含"attention"
                        forward_comm_size *= 2  # 将前向通信大小翻倍
                    # 如果当前层名包含"embedding"
                    if "embedding" in name:  # 检查层名是否包含"embedding"
                        self.workload.append(
                            Work_Item(
                                name=name,  # 工作项名称
                                forward_compute_time=default_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=backward_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )  # 添加嵌入层的工作项
                    else:
                        self.workload.append(
                            Work_Item(
                                name=name,  # 工作项名称
                                forward_compute_time=default_compute_time,  # 前向计算时间
                                forward_comm=forward_comm,  # 前向通信方式
                                forward_comm_size=forward_comm_size,  # 前向通信大小
                                backward_compute_time=default_compute_time,  # 反向计算时间
                                backward_comm=backward_comm,  # 反向通信方式
                                backward_comm_size=backward_comm_size,  # 反向通信大小
                                dp_compute_time=default_compute_time,  # 数据并行计算时间
                                dp_comm=dp_comm,  # 数据并行通信方式
                                dp_comm_size=dp_comm_size,  # 数据并行通信大小
                            )
                        )  # 添加其他层的工作项
        # 添加名为"embedding_norm"的工作项，涉及嵌入层归一化
        self.workload.append(
            Work_Item(
                name="embedding_norm",  # 工作项名称
                forward_compute_time=default_compute_time,  # 前向计算时间
                forward_comm="ALLREDUCE",  # 前向通信方式
                forward_comm_size=self.args.vocab_size * self.args.hidden_size * 2,  # 前向通信大小
                backward_compute_time=default_compute_time,  # 反向计算时间
                backward_comm="NONE",  # 反向通信方式
                backward_comm_size=0,  # 反向通信大小
                dp_compute_time=default_compute_time,  # 数据并行计算时间
                dp_comm="NONE",  # 数据并行通信方式
                dp_comm_size=0,  # 数据并行通信大小
            )
        )  # 添加嵌入归一化的工作项
        # 添加三个"cross_entropy"的工作项
        for i in range(3):  # 循环三次
            self.workload.append(
                Work_Item(
                    name="cross_entropy" + str(i + 1),  # 工作项名称
                    forward_compute_time=compute_time,  # 前向计算时间
                    forward_comm="ALLREDUCE",  # 前向通信方式
                    forward_comm_size=self.args.seq_length * self.args.micro_batch * 4,  # 前向通信大小
                    backward_compute_time=compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )  # 添加"cross_entropy"的工作项
        # 添加四个"optimizer"的工作项
        for i in range(4):  # 循环四次
            self.workload.append(
                Work_Item(
                    name="optimizer" + str(i + 1),  # 工作项名称
                    forward_compute_time=compute_time,  # 前向计算时间
                    forward_comm="ALLREDUCE",  # 前向通信方式
                    forward_comm_size=4,  # 前向通信大小
                    backward_compute_time=compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信方式
                    backward_comm_size=0,  # 反向通信大小
                    dp_compute_time=compute_time,  # 数据并行计算时间
                    dp_comm="NONE",  # 数据并行通信方式
                    dp_comm_size=0,  # 数据并行通信大小
                )
            )  # 添加"optimizer"的工作项

    def dump_file(self, filename):
        # print("??fth def dump_file(self, filen in class simai workload")
        filename = filename + ".txt"  # 将文件名加上.txt后缀

        # 计算管道并行通信值
        pp_comm_value = 2 * self.args.micro_batch * self.args.seq_length * self.args.hidden_size  # 计算管道并行通信大小
        if self.args.enable_sequence_parallel:  # 如果启用序列并行
            pp_comm_value /= self.args.tensor_model_parallel_size  # 将通信大小除以张量模型并行大小

        # 设置pp_comm字符串
        pp_comm = (
            f"pp_comm: {pp_comm_value}"  # 如果管道并行数不为1，设置为具体值
            if self.args.pipeline_model_parallel != 1
            else "pp_comm: 0"  # 否则，设置为0
        )
        with open(filename, "w") as f:  # 打开文件以写入
            # 写入模型和并行配置信息
            f.write((
                f"HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {self.args.tensor_model_parallel_size} "  # 写入模型并行数
                f"ep: {self.args.expert_model_parallel_size} "  # 写入专家模型并行数
                f"pp: {self.args.pipeline_model_parallel} "  # 写入管道并行数
                f"vpp: {self.args.num_layers} "  # 写入垂直并行层数
                f"ga: {self.ga_num} all_gpus: {self.args.world_size} "  # 写入全局批次数和总GPU数
                f"checkpoints: 0 checkpoint_initiates: 0 "  # 写入检查点信息
            ) + pp_comm + "\n")  # 添加管道并行通信信息并换行

            # 写入工作项数量
            f.write(str(len(self.workload)) + "\n")  # 写入工作项总数
            for item in self.workload:  # 遍历每个工作项
                # 将工作项的属性值以制表符分隔后写入文件
                f.write(
                    "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])  # 获取工作项属性并拼接
                    + "\n"  # 换行
                )  # 写入工作项的详细信息


class simAI_MicroTest:  # 定义 simAI_MicroTest 类
    def __init__(self, args):  # 初始化方法，接收参数 args
        self.args = args  # 将传入的 args 赋值给实例变量 self.args
        self.workload = []  # 初始化工作负载列表

    def _simAI_microtest_convert(self, comm_type):  # 定义私有方法 _simAI_microtest_convert，用于转换通信类型
        if comm_type == "all_reduce" or comm_type == "allreduce":  # 如果 comm_type 是 "all_reduce" 或 "allreduce"
            return "ALLREDUCE"  # 返回 "ALLREDUCE"
        elif comm_type == "all_gather" or comm_type == "allgather":  # 如果 comm_type 是 "all_gather" 或 "allgather"
            return "ALLGATHER"  # 返回 "ALLGATHER"
        elif comm_type == "reduce_scatter" or comm_type == "reducescatter":  # 如果 comm_type 是 "reduce_scatter" 或 "reducescatter"
            return "REDUCESCATTER"  # 返回 "REDUCESCATTER"
        elif comm_type == "all_to_all" or comm_type == "alltoall":  # 如果 comm_type 是 "all_to_all" 或 "alltoall"
            return "ALLTOALL"  # 返回 "ALLTOALL"
        else:  # 如果 comm_type 不匹配任何已知类型
            return  # 返回 None

    def workload_generator(self):  # 定义方法 workload_generator，用于生成工作负载
        curr_size = self.args.begin_size  # 初始化当前大小为 begin_size
        default_compute_time = 1  # 设置默认计算时间为 1
        while curr_size <= self.args.end_size:  # 当当前大小小于等于 end_size 时循环
            self.workload.append(  # 向工作负载列表添加一个 Work_Item 实例
                Work_Item(
                    name="micro_test",  # 工作项名称为 "micro_test"
                    forward_compute_time=default_compute_time,  # 前向计算时间
                    forward_comm="NONE",  # 前向通信类型为 "NONE"
                    forward_comm_size=0,  # 前向通信大小为 0
                    backward_compute_time=default_compute_time,  # 反向计算时间
                    backward_comm="NONE",  # 反向通信类型为 "NONE"
                    backward_comm_size=0,  # 反向通信大小为 0
                    dp_compute_time=default_compute_time,  # 数据并行计算时间
                    dp_comm=self._simAI_microtest_convert(self.args.test_comm),  # 数据并行通信类型，通过转换函数获得
                    dp_comm_size=curr_size,  # 数据并行通信大小为当前大小
                    process_time=1,  # 处理时间为 1
                )
            )  # 完成一个工作项的添加
            curr_size *= 2  # 将当前大小翻倍

    def dump_file(self, filename):  # 定义方法 dump_file，用于将工作负载保存到文件
        # print("??fth def dump_file(self, filen in class simai microtest")
        filename = filename + ".txt"  # 文件名添加 ".txt" 后缀
        with open(filename, "w") as f:  # 以写模式打开文件
            if not self.args.multi_all_reduce_enable:  # 如果未启用多重 All Reduce
                f.write(f"MICRO" + "\n")  # 写入 "MICRO" 并换行
                f.write(str(len(self.workload)) + "\n")  # 写入工作负载的数量并换行
                for item in self.workload:  # 遍历每个工作项
                    f.write(  # 写入工作项的详细信息
                        "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])  # 将工作项的属性值以制表符分隔
                        + "\n"  # 添加换行符
                    )  # 完成一个工作项的写入
            else:  # 如果启用了多重 All Reduce
                f.write(  # 写入模型和并行配置信息
                    f"HYBRID_TRANSFORMER_FWD_IN_BCKWD\tmodel_parallel_NPU_group: {self.args.tensor_model_parallel_size} \
                        expert_parallel_npu_group: {self.args.expert_model_parallel_size} pp: {self.args.pipeline_model_parallel} \
                        ga: {self.ga_num} all_gpus: {self.args.world_size} checkpoints: 0 checkpoint_initiates: 0"
                    + "\n"  # 添加换行符
                )  # 完成配置行的写入
                f.write(str(len(self.workload)) + "\n")  # 写入工作负载的数量并换行
                for item in self.workload:  # 遍历每个工作项
                    f.write(  # 写入工作项的详细信息
                        "\t".join([str(getattr(item, k)) for k in item.__dict__.keys()])  # 将工作项的属性值以制表符分隔
                        + "\n"  # 添加换行符
                    )  # 完成一个工作项的写入

if __name__ == "__main__":  # 如果脚本作为主程序运行
    # print(f'-- fth ????? workload_generator/AIOB_simAI_workload_generator_fth.py')
    args = get_params()  # 获取参数
    model = MegatronModel(args)  # 初始化 MegatronModel 模型
    result_dir = "results/workload/"  # 设置结果目录
    if not os.path.isdir(result_dir):  # 如果结果目录不存在
        os.makedirs(result_dir)  # 创建结果目录
    if args.is_inference == False:
        filename = f"{args.gpu_type}-{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-MOE-{args.moe_enable}-GEMM-{args.moe_grouped_gemm}-flash_attn-{args.use_flash_attn}"  # 生成文件名，包含多个参数信息
    elif args.is_inference == True:
        filename = f"is_inference-{args.gpu_type}-{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-MOE-{args.moe_enable}-GEMM-{args.moe_grouped_gemm}-flash_attn-{args.use_flash_attn}"  # 生成文件名，包含多个参数信息
    filepath = os.path.join(result_dir, filename)  # 组合成完整的文件路径
    # print(f"??fth filepath={filepath}")
    params = model.parameters()  # 获取模型的参数
    # 注释掉的工作负载生成和保存步骤
    # work = SIMAI_workload(model, args, GPU_Tensor_core.A100, "gpt13B")  # 初始化 SIMAI_workload 模型
    # name_layers = work.workload_generate()  # 生成工作负载
    # work.dump_file("test")  # 将工作负载保存到文件
    print(sum(p.numel() for p in params))  # 打印模型参数的总数
    if args.aiob_enable:  # 如果启用了 AI-O部门
        params = model.parameters()  # 再次获取模型的参数
        args.model_param = sum(p.numel() for p in params)  # 计算并设置模型参数的总数
        if args.comp_filepath == None:  # 如果计算文件路径未设置
            comp_filepath = get_comp_out(args)  # 获取计算输出文件路径
            compute_cache = extract_averages(comp_filepath, args)  # 提取平均计算值
        else:  # 如果计算文件路径已设置
            print("comp_filepath:", args.comp_filepath)  # 打印计算文件路径
            comp_filepath = args.comp_filepath  # 使用指定的计算文件路径
            compute_cache = extract_averages(comp_filepath, args)  # 提取平均计算值
        print("compute_cache = {")  # 打印 compute_cache 开始
        for key, value in compute_cache.items():  # 遍历 compute_cache 的每个键值对
            print(f"    '{key}' : {value},")  # 打印每个键值对
        print("}")  # 打印 compute_cache 结束
        work = SIMAI_workload(  # 初始化 SIMAI_workload 实例
            model, args, compute_cache  # 传入模型、参数和计算缓存
        )
        if args.is_inference == False:
            name_layers = work.workload_generate_aiob()  # 生成 AI-O部门的工作负载
            work.dump_file(filepath)  # 将工作负载保存到文件
            print("workload save in :", filepath)  # 打印保存路径
        elif args.is_inference == True:
            name_layers = work.workload_generate_aiob_inference()  # 生成 AI-O部门的工作负载
            work.dump_file(filepath)  # 将工作负载保存到文件
            print("workload save in :", filepath)  # 打印保存路径
    # print(args)  # 注释掉的打印参数
    else:  # 如果未启用 AI-O部门
        work = SIMAI_workload(model, args, None)  # 初始化 SIMAI_workload 实例，传入 None 作为计算缓存
        name_layers = work.workload_generate()  # 生成工作负载
        work.dump_file(filepath)  # 将工作负载保存到文件
        print(f"workload save in : {filepath}.txt")  # 打印保存路径

