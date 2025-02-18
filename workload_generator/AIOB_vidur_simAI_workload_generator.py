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
from workload_generator.mocked_model.MockedSarathi import *  # 导入 MockedMegatron 模块的所有内容
from workload_generator.mocked_model.MockedModel import MockedParam, MockedModel  # 导入 MockedParam 和 MockedModel 类
from utils.utils import CommType, get_params, get_comp_out, extract_averages  # 导入工具函数和枚举类型
import os  # 导入操作系统相关模块
from typing import List, Tuple  # 导入类型注解支持
from collections import deque  # 导入双端队列
import dataclasses  # 导入数据类支持
from enum import Enum  # 导入枚举类型支持

try:
    import torch  # 尝试导入 PyTorch 库
except ImportError as e:  # 如果导入失败
    torch = None  # 设置 torch 为 None
    print("Failed to import 'torch'.")  # 打印错误信息
import math  # 导入数学库
import re  # 导入正则表达式库


if __name__ == "__main__":  # 如果脚本作为主程序运行
    args = get_params()  # 获取参数
    model = SarathiModel(args)  # 初始化 MegatronModel 模型
    
    # model = MegatronModel(args)  # 初始化 MegatronModel 模型
    '''
    result_dir = "results/workload/"  # 设置结果目录
    if not os.path.isdir(result_dir):  # 如果结果目录不存在
        os.makedirs(result_dir)  # 创建结果目录
    filename = f"{args.gpu_type}-{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-MOE-{args.moe_enable}-GEMM-{args.moe_grouped_gemm}-flash_attn-{args.use_flash_attn}"  # 生成文件名，包含多个参数信息
    filepath = os.path.join(result_dir, filename)  # 组合成完整的文件路径
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
        name_layers = work.workload_generate_aiob()  # 生成 AI-O部门的工作负载
        work.dump_file(filepath)  # 将工作负载保存到文件
        print("workload save in :", filepath)  # 打印保存路径
    # print(args)  # 注释掉的打印参数
    else:  # 如果未启用 AI-O部门
        work = SIMAI_workload(model, args, None)  # 初始化 SIMAI_workload 实例，传入 None 作为计算缓存
        name_layers = work.workload_generate()  # 生成工作负载
        work.dump_file(filepath)  # 将工作负载保存到文件
        print(f"workload save in : {filepath}.txt")  # 打印保存路径
    '''