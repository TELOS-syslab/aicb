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

from typing import List, Dict  # 导入类型注解工具
import pandas as pd  # 导入pandas库，用于数据处理
import pickle  # 导入pickle模块，用于序列化和反序列化对象
from enum import Enum  # 导入Enum类，用于定义枚举类型
import argparse  # 导入argparse模块，用于解析命令行参数
import sys  # 导入sys模块，用于系统相关操作
import time  # 导入time模块，用于时间相关操作
import os  # 导入os模块，用于文件和目录操作
import json  # 导入json模块，用于JSON数据处理
from collections import defaultdict  # 导入defaultdict，用于创建默认字典
import math  # 导入math模块，用于数学运算
import re  # 导入re模块，用于正则表达式操作

try:
    import torch  # 尝试导入torch库，用于深度学习操作
except ImportError as e:  # 如果导入失败
    torch = None  # 将torch设置为None
    print("Failed to import 'torch'.")  # 打印错误信息


def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool],
) -> List[List[int]]:  # 定义生成正交并行组的函数
    """Generate orthogonal parallel groups based on the parallel size and mask.

    Arguments:
        world_size (int): world size  # 总GPU数量

        parallel_size (List[int]):
            The parallel size of each orthogonal parallel type. For example, if
            tensor_parallel_size = 2, pipeline_model_parallel_group = 3, data_parallel_size = 4,
            and the parallel mapping order is tp-pp-dp, then the parallel_size = [2, 3, 4].

        mask (List[bool]):
            The mask controls which parallel methods the generated groups represent. If mask[i] is
            True, it means the generated group contains the i-th parallelism method. For example, 
            if parallel_size = [tp_size, pp_size, dp_size], and mask = [True, False , True], then 
            the generated group is the `tp-dp` group, if the mask = [False, True, False], then the 
            generated group is the `pp` group.

    Algorithm:
        For orthogonal parallelism, such as tp/dp/pp/cp, the global_rank and
        local_rank satisfy the following equation:
            global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size (1)
                tp_rank \in [0, tp_size)
                dp_rank \in [0, dp_size)
                pp_rank \in [0, pp_size)

        If we want to get the `dp_group` (tp_size * pp_size groups of dp_size ranks each.
        For example,  if the gpu size is 8 and order is 'tp-pp-dp', size is '2-2-2', and the 
        dp_group here is [[0, 4], [1, 5], [2, 6], [3, 7]].)
        The tp_rank and pp_rank will be combined to form the `dp_group_index`.
            dp_group_index = tp_rank + pp_rank * tp_size (2)

        So, Given that tp_rank and pp_rank satisfy equation (2), and dp_rank in
        range(0, dp_size), the ranks in dp_group[dp_group_index] satisfies the
        equation (1).
        
        This function solve this math problem.

    For example, if the parallel_size = [tp_size, dp_size, pp_size] = [2, 3, 4],
    and the mask = [False, True, False]. Then,
        dp_group_index(0) = tp_rank(0) + pp_rank(0) * 2
        dp_group_index(1) = tp_rank(1) + pp_rank(0) * 2
        ...
        dp_group_index(7) = tp_rank(1) + pp_rank(3) * 2

        dp_group[0] = 0 + range(0, 3) * 2 + 0 = [0, 2, 4]
        dp_group[1] = 1 + range(0, 3) * 2 + 0 = [1, 3, 5]
        ...
        dp_group[7] = 1 + range(0, 3) * 2 + 3 * 2 * 3 = [19, 21, 23]
    """

    def prefix_product(a: List[int], init=1) -> List[int]:  # 定义前缀积函数
        r = [init]  # 初始化结果列表
        for v in a:  # 遍历输入列表
            init = init * v  # 计算当前值的累积乘积
            r.append(init)  # 将累积乘积添加到结果列表
        return r  # 返回前缀积列表

    def inner_product(a: List[int], b: List[int]) -> int:  # 定义内积函数
        return sum([x * y for x, y in zip(a, b)])  # 计算两个列表的内积

    def decompose(index, shape, stride=None):  # 定义分解函数
        ''' 
        This function solve the math problem below:
            There is an equation: 
                index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank
        from group_index and rank_in_group.
        '''
        if stride is None:  # 如果未提供步长
            stride = prefix_product(shape)  # 使用前缀积计算步长
        idx = [(index // d) % s for s, d in zip(shape, stride)]  # 分解索引
        # stride是前缀积的结果，stride[-1]的值未使用
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)  # 检查分解是否正确
        return idx  # 返回分解后的索引

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]  # 提取mask为True的形状
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]  # 提取mask为False的形状

    global_stride = prefix_product(parallel_size)  # 计算全局步长
    masked_stride = [d for d, m in zip(global_stride, mask) if m]  # 提取mask为True的步长
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]  # 提取mask为False的步长

    group_size = prefix_product(masked_shape)[-1]  # 计算每个组的大小
    num_of_group = world_size // group_size  # 计算组的数量

    ranks = []  # 初始化结果列表
    for group_index in range(num_of_group):  # 遍历每个组
        # 根据未mask的部分分解组索引
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []  # 初始化当前组的rank列表
        for rank_in_group in range(group_size):  # 遍历组内的每个rank
            # 根据mask的部分分解rank索引
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )  # 计算全局rank
        ranks.append(rank)  # 将当前组的rank列表添加到结果
    return ranks  # 返回所有组的rank列表


class RankGenerator(object):  # 定义Rank生成器类
    def __init__(self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str) -> None:  # 初始化函数
        self.tp = tp  # 张量并行大小
        self.ep = ep  # 专家并行大小
        self.dp = dp  # 数据并行大小
        self.pp = pp  # 流水线并行大小
        self.cp = cp  # 上下文并行大小
        self.world_size = tp * dp * pp * cp  # 计算总GPU数量

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }  # 并行方法名称到大小的映射
        self.order = order  # 并行顺序
        order = order.lower()  # 将顺序转换为小写

        if 'ep' in order:  # 如果顺序中包含专家并行
            if 'ep-dp' not in order and 'dp-ep' not in order:  # 检查专家并行和数据并行是否相邻
                raise RuntimeError(f"The ep and dp must be adjacent in order ({self.order}).")

        for name in self.name_to_size.keys():  # 遍历所有并行方法
            if name not in order and self.name_to_size[name] != 1:  # 如果某方法未指定且大小不为1
                raise RuntimeError(
                    f"The size of ({name}) is ({self.name_to_size[name]}), but you haven't specified the order ({self.order})."
                )
            elif name not in order:  # 如果某方法未指定但大小为1
                order = order + '-' + name  # 将其添加到顺序末尾

        self.order_w_ep = order  # 包含专家并行的顺序
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])  # 不包含专家并行的顺序
        self.ordered_size_wo_ep = []  # 不包含专家并行的大小列表
        self.ordered_size_w_ep = []  # 包含专家并行的大小列表

        for token in order.split('-'):  # 遍历顺序中的每个部分
            if token == 'dp':  # 如果是数据并行
                self.ordered_size_w_ep.append(self.dp // self.ep)  # 添加调整后的数据并行大小
                self.ordered_size_wo_ep.append(self.dp)  # 添加原始数据并行大小
            elif token == 'ep':  # 如果是专家并行
                self.ordered_size_w_ep.append(self.ep)  # 添加专家并行大小
            else:  # 其他并行方法
                self.ordered_size_w_ep.append(self.name_to_size[token])  # 添加对应大小
                self.ordered_size_wo_ep.append(self.name_to_size[token])  # 添加对应大小

    def get_mask(self, order: str, token: str):  # 定义获取mask的函数
        ordered_token = order.split('-')  # 将顺序拆分为列表
        token = token.split('-')  # 将目标token拆分为列表
        mask = [False] * len(ordered_token)  # 初始化mask列表
        for t in token:  # 遍历目标token
            mask[ordered_token.index(t)] = True  # 将对应位置设置为True
        return mask  # 返回mask列表

    def get_ranks(self, token, independent_ep=False):  # 定义获取rank组的函数
        '''Get rank group by input token.

        Arguments:
            token (str):
                Specify the ranks type that want to get. If we want
                to obtain multiple parallel types, we can use a hyphen
                '-' to separate them. For example, if we want to obtain
                the TP_DP group, the token should be 'tp-dp'.

            independent_ep (bool: True):
                This flag controls whether we treat EP and DP independently.
                EP shares ranks with DP, if we want to get ranks related to
                EP, we should set the flag. For example, get_ranks('dp', True)
                will get DP modulo EP group, and get_ranks('dp', False) will
                get full DP group.
        '''
        if independent_ep:  # 如果独立处理专家并行
            parallel_size = self.ordered_size_w_ep  # 使用包含专家并行的大小
            order = self.order_w_ep  # 使用包含专家并行的顺序
        else:  # 否则
            parallel_size = self.ordered_size_wo_ep  # 使用不包含专家并行的大小
            order = self.order_wo_ep  # 使用不包含专家并行的顺序
        mask = self.get_mask(order, token)  # 获取mask
        ranks = generate_masked_orthogonal_rank_groups(self.world_size, parallel_size, mask)  # 生成rank组
        return ranks  # 返回rank组


def gelu_impl(x):  # 定义gelu_impl函数，计算OpenAI的GELU激活
    """OpenAI's gelu implementation."""  # 函数文档字符串，说明这是OpenAI的GELU实现
    return (  # 返回计算结果
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))  # GELU公式实现
    )
    
    
def openai_gelu(x):  # 定义openai_gelu函数
    return gelu_impl(x)  # 调用gelu_impl函数并返回其结果
    
    
def erf_gelu(x):  # 定义erf_gelu函数，另一种GELU实现
    return (  # 返回计算结果
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)  # 计算误差函数
            + torch.ones_like(x).to(dtype=x.dtype)  # 加上全1张量
        )
    )
    
    
def Comp_with_aiob(workload, compute_cache):  # 定义Comp_with_aiob函数，处理计算工作负载
    for item in workload.workload:  # 遍历工作负载中的每个项目
        if item.comm_type == CommType.computation:  # 如果通信类型是计算
            for key in compute_cache:  # 遍历计算缓存中的每个键
                key_temp = key.split("_")[0]  # 分割键名，取第一个部分
                if key_temp in item.stage:  # 如果键名部分在项目阶段中
                    item._elapsed_time = compute_cache[key]  # 设置项目的已用时间
                    break  # 跳出内循环
    return workload  # 返回处理后的工作负载
    
    
def get_comp_out(args):  # 定义get_comp_out函数，获取计算输出
    vocab_size = args.vocab_size  # 获取词汇表大小
    batch_size = args.micro_batch  # 获取微批次大小
    seq_len = args.seq_length  # 获取序列长度
    tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
    vocab_size = args.padded_vocab_size  # 更新词汇表大小为填充后的大小
    if "Megatron" in args.frame:  # 如果框架包含'Megatron'
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        from workload_generator.mocked_model.AiobMegatron import MegatronModel  # 导入MegatronModel

        measure_model = MegatronModel(args)  # 初始化Megatron模型
        measure_model.train()  # 设置模型为训练模式
        if args.dtype == "bfloat16":  # 如果数据类型是bfloat16
            dtype = torch.bfloat16  # 设置dtype为bfloat16
        elif args.dtype == "float16":  # 如果数据类型是float16
            dtype = torch.float16  # 设置dtype为float16
        else:  # 否则
            dtype = torch.float32  # 设置dtype为float32
        # total_input_1 = torch.rand(args.seq_len,  # 注释掉的代码，用于生成随机输入
        #                                       args.batch_size,
        #                                       args.hidden_size,
        #                                       device=device).to(dtype)
        masked_input = torch.randint(  # 生成随机掩码输入
            0,
            math.ceil(vocab_size / tp),  # 范围上限为词汇表大小除以并行大小的向上取整
            (batch_size, seq_len),  # 输入的形状为（批次大小，序列长度）
            device=device,  # 指定设备
            dtype=torch.int64,  # 指定数据类型为int64
        )
        filepath = measure_model(masked_input)  # 运行模型并获取输出文件路径
        return filepath  # 返回文件路径
    
    
def extract_averages(file_path, args):  # 定义extract_averages函数，提取平均值
    attention_avg_sum = 0.0  # 初始化注意力平均和
    mlp_avg_sum = 0.0  # 初始化MLP平均和
    other_avgs = {}  # 初始化其他平均值的字典
    grad_forward = 0.0  # 初始化前向梯度
    grad_backward = 0.0  # 初始化后向梯度

    section_header_re = re.compile(r"^(\w+):")  # 编译部分头的正则表达式
    time_gpu_avg_re = re.compile(r"time_gpu_avg:\s+(\d+(\.\d+)?)")  # 编译GPU平均时间的正则表达式
    time_gpu_min_re = re.compile(r"time_gpu_min:\s+(\d+(\.\d+)?)")  # 编译GPU最小时间的正则表达式

    with open(file_path, "r") as file:  # 打开文件进行读取
        current_section = None  # 初始化当前部分

        for line in file:  # 遍历文件中的每一行
            header_match = section_header_re.match(line)  # 匹配部分头
            if header_match:  # 如果匹配
                current_section = header_match.group(1).strip()  # 设置当前部分名

            avg_match = time_gpu_avg_re.search(line)  # 搜索平均时间匹配
            min_match = time_gpu_min_re.search(line)  # 搜索最小时间匹配
            if current_section == "param_time":  # 如果当前部分是参数时间
                if min_match:  # 如果有最小时间匹配
                    grad_forward = float(min_match.group(1)) * 1000  # 设置前向梯度时间，单位微秒
                if avg_match:  # 如果有平均时间匹配
                    grad_backward = float(avg_match.group(1)) * 1000  # 设置后向梯度时间
            elif avg_match and current_section:  # 如果有平均时间匹配且有当前部分
                avg_value = float(avg_match.group(1)) * 1000  # 计算平均值，单位微秒
                if "atten" in current_section or current_section == "layernorm":  # 如果当前部分是注意力或层归一化
                    if args.recompute_activations and 'flash' in current_section:  # 如果重新计算激活且部分包含'flash'
                        attention_avg_sum += avg_value * 2  # 平均注意力时间增加
                    else:
                        attention_avg_sum += avg_value  # 累加平均注意力时间
                elif "mlp" in current_section or current_section == "layernorm2":  # 如果当前部分是MLP或第二层归一化
                    mlp_avg_sum += avg_value  # 累加MLP平均时间
                else:
                    other_avgs[current_section] = avg_value  # 记录其他部分的平均值

    # 四舍五入并转换为整数
    attention_forward = round(attention_avg_sum)  # 四舍五入注意力前向时间
    attention_backward = attention_forward  # 设置注意力后向时间等于前向时间
    mlp_forward = round(mlp_avg_sum)  # 四舍五入MLP前向时间
    mlp_backward = mlp_forward  # 设置MLP后向时间等于前向时间
    grad_backward = round(grad_backward)  # 四舍五入后向梯度时间
    grad_forward = round(grad_forward)  # 四舍五入前向梯度时间
    other_avgs_int = {k: round(v) for k, v in other_avgs.items() if k != "param_time"}  # 四舍五入其他平均值

    a100_compute_cache = {  # 初始化A100计算缓存字典
        "attention_forward": attention_forward,  # 注意力前向时间
        "attention_backward": attention_backward,  # 注意力后向时间
        "mlp_forward": mlp_forward,  # MLP前向时间
        "mlp_backward": mlp_backward,  # MLP后向时间
        "grad_forward": grad_forward,  # 梯度前向时间
        "grad_backward": grad_backward,  # 梯度后向时间
    }
    a100_compute_cache.update(other_avgs_int)  # 更新其他平均值到计算缓存中

    return a100_compute_cache  # 返回计算缓存
    
    
def process_all_keys(input_file):  # 定义process_all_keys函数，处理所有键
    with open(input_file, "r") as file:  # 打开输入文件进行读取
        first_line_str = file.readline().strip()  # 读取第一行并去除空白
        remaining_content = file.read().strip()  # 读取剩余内容并去除空白
    # 尝试修复和构建合法的 JSON 字符串
    corrected_content = remaining_content.replace("}{", "},{").replace("]}{", "]},{")  # 替换不合法的JSON分隔符

    # 构建 JSON 数组
    json_str = f"[{corrected_content}]"  # 将修正后的内容包裹成JSON数组

    try:
        data = json.loads(json_str)  # 解析JSON字符串

        processed_results = defaultdict(lambda: defaultdict(list))  # 初始化嵌套的默认字典
        for entry in data:  # 遍历数据中的每个条目
            for key, measurements in entry.items():  # 遍历每个键及其测量值
                for measure in measurements:  # 遍历每个测量
                    measure_key, measure_value = next(iter(measure.items()))  # 获取测量的键和值
                    if "time_gpu" in measure_key:  # 如果测量键包含"time_gpu"
                        processed_results[key]["time_gpu"].append(measure["time_gpu"])  # 添加到time_gpu列表
                    else:
                        processed_results[key][measure_key] = measure_value  # 记录其他测量值

        for key, values in processed_results.items():  # 遍历处理后的结果
            if "time_gpu" in values:  # 如果存在time_gpu
                gpu_times = values["time_gpu"]  # 获取所有GPU时间
                min_time_gpu = min(gpu_times)  # 获取最小GPU时间
                gpu_times_filtered = [t for t in gpu_times if t <= 3 * min_time_gpu]  # 过滤时间不超过最小时间三倍的值
                values["time_gpu_max"] = max(gpu_times_filtered)  # 设置最大GPU时间
                values["time_gpu_min"] = min_time_gpu  # 设置最小GPU时间
                values["time_gpu_avg"] = sum(gpu_times_filtered) / len(gpu_times_filtered)  # 计算平均GPU时间
                del values["time_gpu"]  # 删除原始time_gpu列表

        with open(input_file, "w") as outfile:  # 打开输入文件进行写入
            outfile.write(first_line_str + "\n")  # 写入第一行
            for key, values in processed_results.items():  # 遍历处理后的结果
                outfile.write(f"{key}:\n")  # 写入键名
                for subkey, subvalue in values.items():  # 遍历每个子键和子值
                    outfile.write(f"    {subkey}: {subvalue}\n")  # 写入子键和值
        print(f"Compute-results save in:{input_file}")  # 打印保存信息

    except json.JSONDecodeError as e:  # 捕获JSON解析错误
        print(f"Failed to decode JSON: {e}")  # 打印错误信息
        print("Invalid JSON content:\n", corrected_content)  # 打印无效的JSON内容


def cuda_timing_decorator(func):  # 定义cuda_timing_decorator装饰器
    def wrapper(*args, **kwargs):  # 定义包装函数
        start_event = torch.cuda.Event(enable_timing=True)  # 创建开始事件
        end_event = torch.cuda.Event(enable_timing=True)  # 创建结束事件

        start_event.record()  # 记录开始事件
        result = func(*args, **kwargs)  # 调用被装饰的函数并获取结果
        end_event.record()  # 记录结束事件
        torch.cuda.synchronize()  # 同步CUDA

        elapsed_time_ms = start_event.elapsed_time(end_event) * 1000  # 计算经过时间，单位为毫秒
        return result, elapsed_time_ms  # 返回结果和经过时间

    return wrapper  # 返回包装函数


def get_aiob_path(args):  # 定义get_aiob_path函数，获取AIoB结果路径
    result_dir = "./results/aiob_outputs"  # 设置结果目录
    if not os.path.isdir(result_dir):  # 如果目录不存在
        os.makedirs(result_dir)  # 创建目录
    filename = f"{args.model_name}-world_size{args.world_size}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel}-ep{args.expert_model_parallel_size}-gbs{args.global_batch}-mbs{args.micro_batch}-seq{args.seq_length}-flash_attn-{args.use_flash_attn}.txt"  # 构建文件名
    filepath = os.path.join(result_dir, filename)  # 构建文件路径
    return filepath  # 返回文件路径


def write_op(time_list, args):  # 定义write_op函数，写入操作时间列表
    filepath = get_aiob_path(args)  # 获取文件路径
    with open(filepath, "w") as file:  # 打开文件进行写入
        file.write(f"train_iter:{args.epoch_num}\n")  # 写入训练迭代次数
        data_str = json.dumps(time_list, indent=4)  # 将时间列表转换为JSON字符串

        file.write(data_str)  # 写入JSON字符串
    return filepath  # 返回文件路径


class ReduceOp(Enum):  # 定义ReduceOp枚举类
    SUM = 0  # 和
    PRODUCT = 1  # 积
    MIN = 2  # 最小值
    MAX = 3  # 最大值
    BAND = 4  # 位与
    BOR = 5  # 位或
    BXOR = 6  # 位异或
    AVG = 7  # 平均值
    UNUSED = 8  # 未使用


class CommType(str, Enum):  # 定义CommType枚举类，表示通信类型
    """Enum class for possible comm types"""  # 类文档字符串，说明这是通信类型的枚举类

    all_reduce = "all_reduce"  # 全量归约
    isend = "isend"  # 非阻塞发送
    irecv = "irecv"  # 非阻塞接收
    broadcast = "broadcast"  # 广播
    all_gather = "all_gather"  # 全量收集
    reduce_scatter = "reduce_scatter"  # 归约散射
    barrier = "barrier"  # 障碍
    reduce = "reduce"  # 归约
    reduce_scatter_tensor = "reduce_scatter_tensor"  # 归约散射张量
    all_gather_into_tensor = "all_gather_into_tensor"  # 全量收集到张量
    computation = "computation"  # 计算
    epoch_end = "epoch_end"  # 纪元结束
    all_to_all = "all_to_all"  # 全到全

    @classmethod
    def get_comm_type(cls, value):  # 定义类方法get_comm_type，根据值获取通信类型
        for comm_type in cls:  # 遍历所有通信类型
            if comm_type.value == value:  # 如果匹配
                return comm_type  # 返回通信类型
        raise ValueError("Invailid communication type")  # 如果未找到，抛出错误


class CommGroup(str, Enum):  # 定义CommGroup枚举类，表示通信组
    """Enum class for possible comm groups"""  # 类文档字符串，说明这是通信组的枚举类

    dp_group = "dp_group"  # 数据并行组
    pp_group = "pp_group"  # 流水线并行组
    tp_group = "tp_group"  # 张量并行组
    ep_group = "ep_group"  # 专家并行组
    ep_dp_group = "ep_dp_group"  # 专家数据并行组
    ep_tp_group = "ep_tp_group"  # 专家张量并行组
    embedding_group = "embedding_group"  # 嵌入层组
    all = "all_nodes"  # 所有节点组


class WorkloadWriter:  # 定义WorkloadWriter类，用于读写工作负载
    @staticmethod
    def write_workload(workload: List[Dict], args, filename: str):  # 静态方法，写工作负载到文件
        df = pd.DataFrame.from_dict(workload)  # 将工作负载转换为DataFrame
        df = df.fillna(-1)  # 用-1填充缺失值
        df.to_csv(filename, index=False)  # 将DataFrame写入CSV文件

    @staticmethod
    def load_workload(filename: str) -> List[Dict]:  # 静态方法，加载工作负载从文件
        filename = filename.split(".")  # 分割文件名
        filename[-1] = "pkl"  # 替换扩展名为pkl
        filename = ".".join(filename)  # 重新组合文件名
        workload, args = pickle.load(open(filename, "rb"))  # 使用pickle加载数据
        return workload, args  # 返回工作负载和参数
    
    
def get_params():  # 定义get_params函数，解析命令行参数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "--frame",
        help="communication framework",
        choices=["Megatron", "DeepSpeed", "collective_test"],  # 可选框架
        default="Megatron",  # 默认框架
    )
    parser.add_argument("--gpu_type", type=str, default=None),  # 添加gpu_type参数
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of GPUs")  # 添加world_size参数，GPU数量
    parser.add_argument("--tensor_model_parallel_size", type=int, default=1,
                        help='Degree of tensor model parallelism.')  # 添加张量模型并行大小参数
    parser.add_argument("--pipeline_model_parallel", type=int, default=1,
                        help='Degree of pipeline model parallelism.')  # 添加流水线模型并行大小参数
    parser.add_argument('--context-parallel-size', type=int, default=1,
                       help='Degree of context parallelism.')  # 添加上下文并行大小参数
    parser.add_argument("--pp_rank", type=int, default=-1,
                        help='Rank where encoder and decoder should be split.')  # 添加pp_rank参数
    parser.add_argument("--global_batch", type=int, default=4,
                        help='Training batch size. If set, it should be a '
                       'multiple of micro-batch-size times data-parallel-size. '
                       'If this value is None, then '
                       'use micro-batch-size * data-parallel-size as the '
                       'global batch size. This choice will result in 1 for '
                       'number of micro-batches.')  # 添加global_batch参数
    parser.add_argument("--micro_batch", type=int, default=1,
                       help='Batch size per model instance (local batch size). '
                       'Global batch size is local batch size times data '
                       'parallel size times number of micro batches.'  # 添加micro_batch参数
                        )
    parser.add_argument("--epoch_num", type=int, default=1,
                        help="Number of iterations")  # 添加epoch_num参数
    parser.add_argument("--computation_enable", action="store_true", help="Enable computation")  # 添加computation_enable开关
    parser.add_argument("--dtype", default="bfloat16")  # 添加dtype参数，默认bfloat16
    parser.add_argument(
        "--ffn_hidden_size",
        type=int,
        default=None,
        help="Transformer Feed-Forward Network hidden size. "
        "This is set to 4*hidden-size if not provided",
    )  # 添加ffn_hidden_size参数
    parser.add_argument(
        "--enable_visual",
        action="store_true",
        help="Enable visualization",
    )  # 添加enable_visual开关
    parser.add_argument("--workload_only", action="store_true", help="Only generate workload")  # 添加workload_only开关
    get_model_params(parser)  # 添加模型相关参数
    get_ds_params(parser)  # 添加DeepSpeed相关参数
    get_megatron_params(parser)  # 添加Megatron相关参数
    get_collective_test_params(parser)  # 添加集体测试相关参数
    get_moe_params(parser)  # 添加MoE相关参数
    get_simAI_workload_params(parser)  # 添加SimAI工作负载相关参数
    get_aiob_params(parser)  # 添加AIoB相关参数
    args = parser.parse_args()  # 解析参数

    assert (
        args.world_size % (args.tensor_model_parallel_size * args.pipeline_model_parallel) == 0
    ), f"world size: {args.world_size}, tp: {args.tensor_model_parallel_size}, pp: {args.pipeline_model_parallel}"  # 断言world_size可被tp和pp整除
    if args.moe_enable:  # 如果启用MoE
        assert (
            args.moe_enable and args.enable_sequence_parallel
        ), f"moe must be enabled with sequence parallel"  # 断言启用MoE时需要启用序列并行
    args.dp_num = args.world_size // (args.tensor_model_parallel_size * args.pipeline_model_parallel)  # 计算数据并行数量
    # assert args.global_batch % (args.dp_num * args.micro_batch) == 0, \
    #     f"global_batch: {args.global_batch}, dp: {args.dp_num}, micro_batch: {args.micro_batch}"  # 注释掉的断言，确保全局批次可被数据并行和微批次整除
    args.num_microbatches = args.global_batch // (args.dp_num * args.micro_batch)  # 计算微批次数量
    if args.aiob_enable and not args.computation_enable:  # 如果启用AIoB且未启用计算
            args.computation_enable = True  # 自动启用计算
                
    if args.num_attention_heads is None:  # 如果未设置注意力头数量
        args.num_attention_heads = args.num_layers  # 设置为层数

    args.padded_vocab_size = get_padded_vocab_size(args)  # 获取填充后的词汇表大小
    if args.ffn_hidden_size is None:  # 如果未设置FFN隐藏大小
        if args.swiglu:  # 如果使用Swiglu
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64  # 计算Swiglu情况下的FFN隐藏大小

        else:
            args.ffn_hidden_size = 4 * args.hidden_size  # 计算默认情况下的FFN隐藏大小
    if args.swiglu:  # 如果使用Swiglu
        args.gated_linear_unit = True  # 启用门控线性单元
        args.bias_gelu_fusion = False  # 禁用偏置GELU融合
    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:  # 如果专家并行大小大于1
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"  # 断言专家数量不为None
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."  # 断言专家数量可被专家并行大小整除
        assert not args.dtype == "float16", \
            "Expert parallelism is not supported with fp16 training."  # 断言不使用float16训练
    if args.moe_grouped_gemm:  # 如果启用MoE分组GEMM
        assert args.dtype == "bfloat16", 'Currently GroupedGEMM for MoE only supports bf16 dtype.'  # 断言数据类型为bfloat16
    if args.pipeline_model_parallel > 1 :  # 如果流水线并行大小大于1
        args.num_layers = int(args.num_layers//args.pipeline_model_parallel)  # 调整层数
    return args  # 返回解析后的参数


ARGS = None  # 初始化全局ARGS变量
    
    
def get_args():  # 定义get_args函数，获取参数
    global ARGS  # 声明使用全局ARGS
    if ARGS is not None:  # 如果ARGS已存在
        return ARGS  # 返回ARGS
    ARGS = get_params()  # 调用get_params获取参数
    return ARGS  # 返回ARGS
    
    
def get_aiob_params(parser: argparse.ArgumentParser):  # 定义get_aiob_params函数，添加AIoB相关参数
    parser.add_argument(
        "--aiob_enable",
        action="store_true",
        help="Enable aiob to get operation real compute time",  # 启用AIoB以获取操作实际计算时间
    )
    parser.add_argument("--comp_filepath", type=str, default=None,
                        help="Use aiob_lib to get operation real compute time",)  # 设置计算文件路径
    parser.add_argument("--gated_linear_unit", default=False)  # 启用门控线性单元
    parser.add_argument("--bias_gelu_fusion", action="store_true",
                        help='Enable bias and gelu fusion.')  # 启用偏置和GELU融合
    parser.add_argument("--openai_gelu", action="store_true",
                         help='Use OpenAIs GeLU implementation. This option'
                       'should not be used unless for backward compatibility'
                       'reasons.')  # 使用OpenAI的GELU实现，仅为兼容性目的
    parser.add_argument("--onnx_safe", action="store_true",
                        help='Use workarounds for known problems with '
                       'Torch ONNX exporter')  # 启用ONNX安全工作
    parser.add_argument("--squared_relu", action="store_true",
                        help='Use squared relu activation instead of default gelu')  # 使用平方ReLU激活
    parser.add_argument('--recompute_activations', action='store_true',
                       help='recompute activation to allow for training '
                       'with larger models, sequences, and batch sizes.')  # 重新计算激活以允许训练更大模型


def get_model_params(parser: argparse.ArgumentParser):  # 定义get_model_params函数，添加模型相关参数
    parser.add_argument("--model_name", help="Model for training")  # 添加model_name参数
    parser.add_argument(
        "--hidden_size", type=int, help='Tansformer hidden size.', default=1024  # 添加hidden_size参数，默认1024
    )
    parser.add_argument("--num_layers", type=int, help='Number of transformer layers.', default=24)  # 添加num_layers参数，默认24
    parser.add_argument(
        "--seq_length", type=int, help='Maximum sequence length to process.', default=2048  # 添加seq_length参数，默认2048
    )
    parser.add_argument("--num_attention_heads", help='Number of transformer attention heads.',type=int, default=None)  # 添加num_attention_heads参数
    parser.add_argument("--vocab_size", type=int, help='Size of vocab before EOD or padding.', default=32000)  # 添加vocab_size参数，默认32000
    parser.add_argument("--max_position_embeddings", type=int,help='Maximum number of position embeddings to use. '
                       'This is the size of position embedding.', default=4096)  # 添加max_position_embeddings参数，默认4096
    parser.add_argument("--add_bias_linear",help='Enable bias in the linear layers', action="store_true")  # 启用线性层中的偏置
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="Use FlashAttention implementation of attention.",  # 使用FlashAttention实现注意力
    )
    parser.add_argument(
        "--swiglu",
        action="store_true",
        help="Use gated linear units and SiLU activation instead of default gelu",  # 使用门控线性单元和SiLU激活代替默认GELU
    )


def get_ds_params(parser: argparse.ArgumentParser):  # 定义get_ds_params函数，添加DeepSpeed相关参数
    parser.add_argument("--stage", type=int, default=3, choices=[1, 2, 3])  # 添加stage参数，选择1、2或3
    parser.add_argument("--amp_enabled", action="store_true")  # 启用自动混合精度
    parser.add_argument("--reduce_bucket_size", type=int, default=int(5e8))  # 添加reduce_bucket_size参数，默认5e8

    # for stage1/2 only
    parser.add_argument("--allgather_bucket_size", type=int, default=int(5e8))  # 添加allgather_bucket_size参数，默认5e8
    parser.add_argument("--contiguous_gradients", action="store_true")  # 启用连续梯度

    # for stage 3 only
    parser.add_argument("--param_persistence_threshold", type=int, default=int(1e5))  # 添加param_persistence_threshold参数，默认1e5
    parser.add_argument(
        "--model_persistence_threshold", type=int, default=int(sys.maxsize)  # 添加model_persistence_threshold参数，默认系统最大值
    )
    parser.add_argument("--max_live_parameters", type=int, default=int(1e9))  # 添加max_live_parameters参数，默认1e9
    parser.add_argument("--prefetch_bucket_size", type=int, default=int(1e9))  # 添加prefetch_bucket_size参数，默认1e9


def get_megatron_params(parser: argparse.ArgumentParser):  # 定义get_megatron_params函数，添加Megatron相关参数
    parser.add_argument("--enable_sequence_parallel",help='Enable sequence parallel optimization.',action="store_true")  # 启用序列并行优化
    parser.add_argument(
        "--use-distributed-optimizer",
        action="store_true",
        help="Use distributed optimizer.",  # 使用分布式优化器
    )
    parser.add_argument("--make_vocab_size_divisible_by", help='Pad the vocab size to be divisible by this value.'
                       'This is added for computational efficieny reasons.',type=int, default=128)  # 设置词汇表大小填充，使其可被指定值整除，默认128
    parser.add_argument(
        "--overlap_grad_reduce",
        action="store_true",
        default=False,
        help="If set, overlap DDP grad reduce. (Not implement yet)",  # 如果设置，重叠DDP梯度归约（尚未实现）
    )
    
    
def get_collective_test_params(parser: argparse.ArgumentParser):  # 定义get_collective_test_params函数，添加集体测试相关参数
    parser.add_argument("--begin_size", type=int, default=1048576)  # 添加begin_size参数，默认1048576
    parser.add_argument("--end_size", type=int, default=1048576)  # 添加end_size参数，默认1048576
    parser.add_argument("--test_comm", type=str, default="all_reduce")  # 添加test_comm参数，默认"all_reduce"
    parser.add_argument("--iter_num", type=int, default=500)  # 添加iter_num参数，默认500
    parser.add_argument("--multi_all_reduce_enable", type=int, default=0)  # 添加multi_all_reduce_enable参数，默认0
    
    
def get_simAI_workload_params(parser: argparse.ArgumentParser):  # 定义get_simAI_workload_params函数，添加SimAI工作负载相关参数
    parser.add_argument("--overlap_version", action="store_true")  # 添加overlap_version开关


def get_moe_params(parser: argparse.ArgumentParser):  # 定义get_moe_params函数，添加MoE相关参数
    parser.add_argument('--moe_enable', action="store_true")  # 启用MoE
    parser.add_argument('--expert_model_parallel_size', type=int, default=1, help='Degree of expert model parallelism.')  # 设置专家模型并行大小，默认1
    parser.add_argument('--num_experts', type=int, default=1, help='Number of Experts in MoE (None means no MoE)')  # 设置MoE中的专家数量，默认1
    parser.add_argument('--moe_router_topk', type=int, default=1, help='Number of experts to route to for each token. The default is 2.')  # 设置每个token路由的专家数量，默认2
    parser.add_argument('--moe_grouped_gemm', action='store_true',
                       help='When there are multiple experts per rank, compress multiple local (potentially small) gemms in a single kernel launch to improve the utilization and performance by leveraging the Grouped GEMM feature introduced since CUTLASS 2.8 (https://github.com/fanshiqing/grouped_gemm).')  # 启用MoE分组GEMM以提高性能
    parser.add_argument('--activation_func', type=str, help='activation_func for mlp')  # 设置MLP的激活函数
    

def ensure_divisibility(numerator, denominator):  # 定义ensure_divisibility函数，确保可除性
    """Ensure that numerator is divisible by the denominator."""  # 函数文档字符串
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )  # 断言分子可被分母整除，若不行则抛出错误
    

def get_padded_vocab_size(args):  # 定义get_padded_vocab_size函数，获取填充后的词汇表大小
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""  # 函数文档字符串，说明填充词汇表大小的目的

    after = args.vocab_size  # 初始化后续大小为词汇表大小

    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size  # 计算需要的倍数
    while (after % multiple) != 0:  # 当当前大小不可被倍数整除时
        after += 1  # 增加大小

    return after  # 返回填充后的大小
    

def divide(numerator, denominator):  # 定义divide函数，执行除法
    """Ensure that numerator is divisible by the denominator and return
    the division value."""  # 函数文档字符串
    ensure_divisibility(numerator, denominator)  # 确保可除性
    return numerator // denominator  # 返回整除结果

