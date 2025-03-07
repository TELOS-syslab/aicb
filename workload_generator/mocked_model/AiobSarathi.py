"""
Copyright (c) 2021, Alibaba Group;  # 版权所有 (c) 2021，阿里巴巴集团；
Licensed under the Apache License, Version 2.0 (the "License");  # 根据Apache许可证版本2.0（“许可证”）授权；
you may not use this file except in compliance with the License.  # 除非符合许可证规定，否则您不得使用此文件。
You may obtain a copy of the License at  # 您可以在以下位置获取许可证副本：
   http://www.apache.org/licenses/LICENSE-2.0  # http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software  # 除非适用法律要求或书面同意，否则
distributed under the License is distributed on an "AS IS" BASIS,  # 根据许可证分发的软件是按“原样”分发的，
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不附带任何明示或暗示的担保或条件。
See the License for the specific language governing permissions and  # 请参阅许可证以了解管理权限和
limitations under the License.  # 许可证下的限制。
"""

import torch  # 导入PyTorch库
import time  # 导入时间模块
import warnings  # 导入警告模块
import torch.nn.functional as F  # 从PyTorch中导入神经网络功能模块
from apex.contrib.layer_norm.layer_norm import FastLayerNormFN  # 从APEX库中导入快速层归一化函数
import math  # 导入数学模块
import scaled_upper_triang_masked_softmax_cuda  # 导入自定义CUDA模块
from torch.cuda.amp import custom_bwd, custom_fwd  # 从PyTorch中导入混合精度训练的自定义前向和反向函数
from utils.utils import *  # 从utils模块中导入所有内容
from core import grouped_gemm_util as gg  # 从core模块中导入grouped_gemm_util，并重命名为gg
try:
    from einops import rearrange  # 尝试从einops库中导入rearrange函数
except ImportError as e:
    rearrange = None  # 如果导入失败，将rearrange设置为None
    print("Failed to import 'einops'. Functions using 'rearrange' might not work.")  # 打印导入失败的警告信息
from typing import Callable, Optional  # 从typing模块中导入Callable和Optional类型

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func  # 尝试从flash_attn库中导入flash_attn_unpadded_func函数
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_unpadded_func,  # 尝试导入flash_attn_varlen_func并重命名为flash_attn_unpadded_func
        )
    except ImportError:
        flash_attn_unpadded_func = None  # 如果仍然导入失败，将flash_attn_unpadded_func设置为None


# class MegatronModel(torch.nn.Module):  # 定义MegatronModel类，继承自torch.nn.Module
class SarathiModel(torch.nn.Module):  # 定义MegatronModel类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数

        # super(MegatronModel, self).__init__()  # 调用父类的初始化方法
        super(SarathiModel, self).__init__()  # 调用父类的初始化方法
        self.time_list = {}  # 初始化time_list为空字典，用于记录时间
        self.args = args  # 保存传入的参数

        # self.Embedding = MegatronEmbedding(self.args)  # 初始化嵌入层
        # self.Layernorm = MegatronLayernorm(self.args)  # 初始化层归一化层
        self.Embedding = SarathiEmbedding(self.args)  # 初始化嵌入层
        self.Layernorm = SarathiLayernorm(self.args)  # 初始化层归一化层
        if self.args.use_flash_attn:  # 如果使用flash attention
            # self.Attention = MegatronFlashAtten(self.args)  # 使用MegatronFlashAtten
            self.Attention = SarathiFlashAtten(self.args)  # 使用MegatronFlashAtten
        else:
            # self.Attention = MegatronAtten(self.args)  # 否则使用MegatronAtten
            self.Attention = SarathiAtten(self.args)  # 否则使用MegatronAtten
        if self.args.moe_enable:  # 如果启用MoE（专家模型）
            self.Mlp = MoELayer(self.args)  # 使用MoELayer
        else:
            # self.Mlp = MegatronMlp(self.args)  # 否则使用MegatronMlp
            self.Mlp = SarathiMlp(self.args)  # 否则使用MegatronMlp
        self.logit = logit(self.args)  # 初始化logit层
        self.grad_param = Grad_param(self.args)  # 初始化梯度参数
        
        # print(">>fth 调用我啦SarathiModel init /disk1/futianhao/software1/aicb/workload_generator/mocked_model/AiobSarathi.py")

    def forward(self, input):  # 前向传播方法，接受输入
        # print(">>fth 调用我啦SarathiModel forward /disk1/futianhao/software1/aicb/workload_generator/mocked_model/AiobSarathi.py")

        # if self.args.warm_up:
        #     for _ in range(10):
        #
        #         layernorm = self.Layernorm._apply()
        #         atten_qkv = self.Attention._apply_attenqkv()
        #         if self.args.use_flash_attn :
        #             atten_core = self.Attention._apply_flash_atten()
        #         else:
        #             atten_core_qk = self.Attention._apply_QK()
        #             atten_core_softmax = self.Attention._apply_Softmax()
        #             atten_core_contex = self.Attention._apply_Contex()
        #         atten_linear = self.Attention._apply_Linear()
        #         layernorm2 = self.Layernorm._apply()
        #         mlp_linear_1 = self.Mlp._apply_Linear1()
        #         mlp_gelu = self.Mlp._apply_activation()
        #         mlp_linear_2 = self.Mlp._apply_Linear2()

        for _ in range(self.args.epoch_num):  # 循环epoch_num次
            # #Embedding
            Emb_output, Emb_time = self.Embedding(input)  # 通过嵌入层处理输入，得到输出和时间
            self.time_list.setdefault("Emb", []).append({"time_gpu": Emb_time})  # 记录嵌入层的GPU时间

            for _ in range(self.args.num_layers):  # 循环num_layers层
                # #layernorm
                lay_out, layernorm = self.Layernorm(Emb_output)  # 通过层归一化处理嵌入输出，得到输出和时间
                self.time_list.setdefault("layernorm", []).append(
                    {"time_gpu": layernorm}  # 记录层归一化的GPU时间
                )
                if self.args.use_flash_attn:  # 如果使用flash attention
                    atten_output, atten_qkv, atten_core, atten_linear = self.Attention(
                        lay_out  # 通过注意力层处理归一化输出，得到多个输出
                    )
                    self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv}  # 记录注意力QKV的GPU时间
                    )
                    self.time_list.setdefault("atten_flash", []).append(
                        {"time_gpu": atten_core}  # 记录闪存注意力核心的GPU时间
                    )
                    self.time_list.setdefault("atten_linear", []).append(
                        {"time_gpu": atten_linear}  # 记录注意力线性层的GPU时间
                    )
                else:
                    (
                        atten_output,
                        atten_qkv,
                        atten_core_qk,
                        atten_core_softmax,
                        atten_core_contex,
                        atten_linear,
                    ) = self.Attention(lay_out)  # 通过注意力层处理，得到多个具体部分的输出
                    self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv}  # 记录注意力QKV的GPU时间
                    )
                    self.time_list.setdefault("atten_core_qk", []).append(
                        {"time_gpu": atten_core_qk}  # 记录注意力QK核心的GPU时间
                    )
                    self.time_list.setdefault("atten_core_softmax", []).append(
                        {"time_gpu": atten_core_softmax}  # 记录注意力Softmax核心的GPU时间
                    )
                    self.time_list.setdefault("atten_core_contex", []).append(
                        {"time_gpu": atten_core_contex}  # 记录注意力上下文核心的GPU时间
                    )
                    self.time_list.setdefault("atten_linear", []).append(
                        {"time_gpu": atten_linear}  # 记录注意力线性层的GPU时间
                    )
                # layernorm
                lay2_out, layernorm2 = self.Layernorm(atten_output)  # 再次通过层归一化处理注意力输出，得到输出和时间

                # mlp layer
                mlp_out, mlp_linear_1, mlp_gelu, mlp_linear_2 = self.Mlp(lay2_out)  # 通过MLP层处理，得到多个输出
                self.time_list.setdefault("layernorm2", []).append(
                    {"time_gpu": layernorm2}  # 记录第二次层归一化的GPU时间
                )
                self.time_list.setdefault("mlp_linear_1", []).append(
                    {"time_gpu": mlp_linear_1}  # 记录MLP第一线性层的GPU时间
                )
                self.time_list.setdefault("mlp_gelu", []).append({"time_gpu": mlp_gelu})  # 记录MLP GELU激活的GPU时间
                self.time_list.setdefault("mlp_linear_2", []).append(
                    {"time_gpu": mlp_linear_2}  # 记录MLP第二线性层的GPU时间
                )

            lay_post__out, layernorm_post = self.Layernorm(mlp_out)  # 最后通过层归一化处理MLP输出，得到输出和时间
            self.time_list.setdefault("layernorm_post", []).append(
                {"time_gpu": layernorm_post}  # 记录后层归一化的GPU时间
            )
            logit_out, logit_time = self.logit(lay_post__out)  # 通过logit层处理，得到输出和时间
            self.time_list.setdefault("logit_time", []).append({"time_gpu": logit_time})  # 记录logit层的GPU时间
            _, param_time = self.grad_param._apply()  # 应用梯度参数，得到时间

            self.time_list.setdefault("param_time", []).append({"time_gpu": param_time})  # 记录参数时间

        filepath = write_op(self.time_list, self.args)  # 写入操作记录到文件
        process_all_keys(filepath)  # 处理所有键
        return filepath  # 返回文件路径


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):  # 定义自定义自动梯度函数类
    """See linear_with_grad_accumulation_and_async_allreduce"""  # 参考linear_with_grad_accumulation_and_async_allreduce

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        tp,
    ):  # 前向传播静态方法，接受多个参数
        ctx.save_for_backward(input, weight)  # 保存输入和权重以备反向传播使用
        ctx.use_bias = bias is not None  # 记录是否使用偏置
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion  # 记录是否使用梯度累积融合
        ctx.async_grad_allreduce = async_grad_allreduce  # 记录是否异步梯度全归约
        ctx.sequence_parallel = sequence_parallel  # 记录是否使用序列并行

        if sequence_parallel:  # 如果使用序列并行
            total_input = input  # 总输入为输入
        else:
            total_input = input  # 否则，总输入也为输入

        output = torch.matmul(total_input, weight.t())  # 计算矩阵乘法

        if bias is not None:  # 如果有偏置
            output = output + bias  # 加上偏置
        return output  # 返回输出


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    tp,
) -> torch.Tensor:  # 定义线性层与梯度累积和异步通信的函数
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.
    # 文档字符串，描述函数的作用

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.
    # 解释梯度累积融合的功能

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.
    # 解释异步梯度全归约的功能

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calculation of the
    weight gradients.
    # 解释序列并行情况下的梯度处理方式

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.
    # 解释使用此模块的环境变量要求

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion.

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel is True, this must be
        False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        tp,
    ]  # 将所有参数打包成列表

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)  # 应用自定义自动梯度函数


linear_with_grad_accumulation_and_async_allreduce.warned = False  # 设置警告标志为False


# Embedding
# class MegatronEmbedding(torch.nn.Module):  # 定义MegatronEmbedding类，继承自torch.nn.Module
class SarathiEmbedding(torch.nn.Module):  # 定义MegatronEmbedding类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        # super(MegatronEmbedding, self).__init__()  # 调用父类的初始化方法
        super(SarathiEmbedding, self).__init__()  # 调用父类的初始化方法
        self.tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度
        hidden_size = args.hidden_size  # 获取隐藏层大小
        max_position_embeddings = args.max_position_embeddings  # 获取最大位置嵌入
        self.vocab_size = args.padded_vocab_size  # 获取填充后的词汇表大小
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            self.dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            self.dtype = torch.float16  # 设置数据类型为float16
        else:
            self.dtype = torch.float32  # 否则设置为float32

        # self.masked_input = torch.randint(0,math.ceil(self.vocab_size/self.tp),
        #                                   (micro_batch,seq_len),
        #                                   device=device, dtype=torch.int64)
        self.weight = torch.randint(
            0,
            1,
            (math.ceil(self.vocab_size / self.tp), hidden_size),
            device=device,
            dtype=torch.int64,
        )  # 初始化权重为随机整数
        self.position_embeddings = torch.nn.Embedding(
            max_position_embeddings, hidden_size
        ).to(device)  # 初始化位置嵌入层并移动到指定设备
        self.position_ids = torch.randint(
            0, seq_len - 1, (1, seq_len), device=device, dtype=torch.int64
        )  # 随机生成位置ID

    @cuda_timing_decorator
    def _apply(self, input):  # 自定义应用方法，接受输入
        # words_embeddings = F.embedding(self.masked_input,
        #                                self.weight,
        #                                 None, None,
        #                                 2.0, False,
        #                                 False)
        # input_ = input

        if self.tp > 1:  # 如果张量并行大小大于1
            # Build the mask.
            input_mask = (input < 0) | (input >= math.ceil(self.vocab_size / self.tp))  # 构建输入掩码
            # Mask the input.
            masked_input = input.clone() - 0  # 克隆输入并减去0
            masked_input[input_mask] = 0  # 将掩码位置设置为0

        else:
            masked_input = input  # 否则，掩码输入为原输入
        words_embeddings = self.weight[masked_input]  # 获取词嵌入

        position_embeddings_i = self.position_embeddings(self.position_ids)  # 获取位置嵌入

        embeddings = words_embeddings + position_embeddings_i  # 词嵌入和位置嵌入相加
        embeddings = embeddings.transpose(0, 1).contiguous()  # 转置并确保内存连续

        return embeddings  # 返回嵌入

    def forward(self, input):  # 前向传播方法，接受输入
        result, emb_time = self._apply(input)  # 通过_apply方法处理输入，得到结果和时间

        result = result.to(self.dtype)  # 将结果转换为指定的数据类型

        return result, emb_time  # 返回结果和时间

# class MegatronLayernorm(torch.nn.Module):  # 定义MegatronLayernorm类，继承自torch.nn.Module
class SarathiLayernorm(torch.nn.Module):  # 定义MegatronLayernorm类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        # super(MegatronLayernorm, self).__init__()  # 调用父类的初始化方法
        super(SarathiLayernorm, self).__init__()  # 调用父类的初始化方法
        self.tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        self.enable_sequence_parallel = args.enable_sequence_parallel  # 获取是否启用序列并行
        hidden_size = args.hidden_size  # 获取隐藏层大小
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            self.dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            self.dtype = torch.float16  # 设置数据类型为float16
        else:
            self.dtype = torch.float32  # 否则设置为float32
        # self.input_l = torch.rand(seq_len,
        #                           micro_batch,
        #                           hidden_size,
        #                           device=device).to(dtype)
        self.lay_weight = torch.rand(hidden_size, device=device).to(self.dtype)  # 初始化层归一化权重为随机值
        self.bias = torch.zeros(hidden_size, device=device).to(self.dtype)  # 初始化偏置为零

    @cuda_timing_decorator  # 使用CUDA计时装饰器 采集FastLayerNormFN层的计算时间
    def _apply(self, hidden_states):  # 定义_apply方法，接受隐藏状态作为输入
        output_lay = FastLayerNormFN.apply(
            hidden_states, self.lay_weight, self.bias, 1e-05  # 应用快速层归一化函数
        )

        return output_lay  # 返回归一化后的输出

    def forward(self, hidden_states):  # 定义前向传播方法，接受隐藏状态作为输入
        # hidden_states = hidden_states.to(self.dtype)
        if self.enable_sequence_parallel:  # 如果启用序列并行
            chunks = torch.chunk(hidden_states, self.tp, 0)  # 将隐藏状态沿第0维切分为tp块
            hidden_states = chunks[0]  # 取第一块作为隐藏状态

        lay_out, lay_time = self._apply(hidden_states)  # 应用归一化方法，@cuda_timing_decorator 获取输出和时间
        if self.enable_sequence_parallel:  # 如果启用序列并行
            lay_out = lay_out.repeat((self.tp, 1, 1))  # 重复归一化输出tp次

        return lay_out, lay_time  # 返回归一化输出和时间


# class MegatronAtten(torch.nn.Module):  # 定义MegatronAtten类，继承自torch.nn.Module
class SarathiAtten(torch.nn.Module):  # 定义MegatronAtten类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        # super(MegatronAtten, self).__init__()  # 调用父类的初始化方法
        super(SarathiAtten, self).__init__()  # 调用父类的初始化方法
        self.enable_sequence_parallel = args.enable_sequence_parallel  # 获取是否启用序列并行
        self.tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度
        hidden_size = args.hidden_size  # 获取隐藏层大小
        num_attention_heads = args.num_attention_heads  # 获取注意力头数量
        self.input_in_float16 = False  # 初始化输入是否为float16的标志
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            dtype = torch.float16  # 设置数据类型���float16
            self.input_in_float16 = True  # 设置输入为float16的标志
        else:
            dtype = torch.float32  # 否则设置为float32
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        # self.atten_total_input_1 = torch.rand(seq_len,
        #                                       micro_batch,
        #                                       hidden_size,
        #                                       device=device).to(dtype)
        self.atten_weight_1 = torch.rand(
            divide((3 * hidden_size), self.tp), hidden_size, device=device
        ).to(dtype)  # 初始化注意力权重为随机值，并根据并行大小分割
        self.hidden_size_per_partition = divide(hidden_size, self.tp)  # 计算每个分区的隐藏层大小
        self.num_attention_heads_per_partition = divide(num_attention_heads, self.tp)  # 计算每个分区的注意力头数量
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)  # 计算每个注意力头的隐藏层大小
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition  # 设置每个分区的查询组数量
        query_layer = torch.rand(
            seq_len,
            micro_batch,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            device=device,
        ).to(dtype)  # 初始化查询层为随机值
        key_layer = query_layer  # 设置键层等于查询层
        value_layer = query_layer  # 设置值层等于查询层
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )  # 定义输出尺寸
        self.query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )  # 重塑查询层
        self.key_layer = key_layer.view(
            output_size[3], output_size[0] * output_size[1], -1
        )  # 重塑键层
        self.matmul_input_buffer = torch.zeros(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            device=device,
        ).to(dtype)  # 初始化矩阵乘法输入缓冲区为零
        self.scale_t = torch.tensor(1).to(dtype)  # 初始化缩放因子
        soft_input = torch.rand(output_size, device=device).to(dtype)  # 初始化softmax输入为随机值
        self.b, self.np, self.sq, self.sk = soft_input.size()  # 获取softmax输入的尺寸
        self.soft_input_1 = soft_input.view(-1, self.sq, self.sk)  # 重塑softmax输入
        self.output_size_2 = (
            value_layer.size(1),
            value_layer.size(2),
            self.query_layer.size(0),
            value_layer.size(3),
        )  # 定义第二次输出尺寸
        self.value_layer = value_layer.view(
            value_layer.size(0), self.output_size_2[0] * self.output_size_2[1], -1
        )  # 重塑值层
        self.atten_linear_weight = torch.rand(
            hidden_size, self.hidden_size_per_partition, device=device
        ).to(dtype)  # 初始化注意力线性权重为随机值
        # self.linear_function = LinearWithGradAccumulationAndAsyncCommunication.apply

    def get_batch_per_block(self, sq, sk, b, np):  # 定义获取每块批次数的方法
        import scaled_masked_softmax_cuda  # 导入自定义CUDA模块

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)  # 调用CUDA模块的方法

    def is_kernel_available(self, b, np, sq, sk):  # 定义检查内核是否可用的方法
        attn_batches = b * np  # 计算注意力批次数

        if (
            self.input_in_float16  # 输入必须是fp16
            and 16 < sk <= 16384  # sk 必须在16到16384之间
            and sq % 4 == 0  # sq 必须是4的倍数
            and sk % 4 == 0  # sk 必须是4的倍数
            and attn_batches % 4 == 0  # np * b 必须是4的倍数
        ):
            if 0 <= sk <= 16384:  # 如果sk在0到16384之间
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)  # 获取每块批次数

                if attn_batches % batch_per_block == 0:  # 如果注意力批次数能被每块批次数整除
                    return True  # 内核可用

        return False  # 否则内核不可用

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_attenqkv(self, hideen_states):  # 定义_apply_attenqkv方法，接受隐藏状态作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数
        output = _forward_impl(
            input=hideen_states,  # 输入隐藏状态
            weight=self.atten_weight_1,  # 使用注意力权重1
            bias=None,  # 无偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=False,  # 禁用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算
        # linear_function = LinearWithGradAccumulationAndAsyncCommunication.apply
        # output = linear_function(self.atten_total_input_1,self.atten_weight_1,None,None,False,False)
        # # output = torch.matmul(self.atten_total_input_1, self.atten_weight_1)
        new_tensor_shape = output.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )  # 计算新的张量形状
        output = output.view(*new_tensor_shape)  # 重塑输出张量
        (query_layer, key_layer, value_layer) = torch.split(
            output,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )  # 分割输出为查询、键、值层
        return query_layer, key_layer, value_layer  # 返回查询、键、值层

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_QK(self, q, k):  # 定义_apply_QK方法，接受查询和键作为输入
        matmul_result = torch.baddbmm(
            self.matmul_input_buffer,  # 使用预先定义的输入缓冲区
            q.transpose(0, 1),  # 转置查询
            k.transpose(0, 1).transpose(1, 2),  # 转置键
            beta=0.0,  # 设置beta参数
            alpha=(1.0 / 11.313708498984761),  # 设置alpha参数
        )  # 执行批量加法矩阵乘法
        return matmul_result  # 返回矩阵乘法结果

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Softmax(self, attention_scores):  # 定义_apply_Softmax方法，接受注意力分数作为输入
        if self.is_kernel_available(*attention_scores.size()):  # 如果内核可用
            b, np, sq, sk = attention_scores.size()  # 获取注意力分数的尺寸
            attention_scores = attention_scores.view(-1, sq, sk)  # 重塑注意力分数
            softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
                attention_scores, self.scale_t  # 使用自定义CUDA软最大函数
            )
            prob = softmax_results.view(self.b, self.np, self.sq, self.sk)  # 重塑软最大结果
        else:
            if self.scale_t is not None:  # 如果存在缩放因子
                attention_scores = attention_scores * self.scale_t  # 缩放注意力分数
            prob = torch.nn.Softmax(dim=-1)(attention_scores)  # 使用PyTorch的软最大函数

        return prob  # 返回概率

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Contex(self, prob, value_layer):  # 定义_apply_Contex方法，接受概率和价值层作为输入
        value_layer = value_layer.view(
            value_layer.size(0), self.output_size_2[0] * self.output_size_2[1], -1
        )  # 重塑值层
        attention_probs = prob.view(
            self.output_size_2[0] * self.output_size_2[1], self.output_size_2[2], -1
        )  # 重塑注意力概率
        output = torch.bmm(attention_probs, value_layer.transpose(0, 1))  # 执行批量矩阵乘法
        context_layer = (
            output.view(*self.output_size_2).permute(2, 0, 1, 3).contiguous()  # 重塑并转置输出为上下文层
        )
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )  # 计算新的上下文层形状
        context_layer = context_layer.view(*new_context_layer_shape)  # 重塑上下文层
        return context_layer  # 返回上下文层

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Linear(self, context_layer):  # 定义_apply_Linear方法，接受上下文层作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数
        output_parallel = _forward_impl(
            input=context_layer,  # 输入上下文层
            weight=self.atten_linear_weight,  # 使用注意力线性权重
            bias=None,  # 无偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=False,  # 禁用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算
        return output_parallel  # 返回并行输出

    def forward(self, hideen_states):  # 定义前向传播方法，接受隐藏状态作为输入
        qkv_out, qkv_time = self._apply_attenqkv(hideen_states)  # 获取查询、键、值层输出和时间

        query_layer, key_layer, value_layer = qkv_out  # 解包查询、键、值层

        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )  # 定义输出尺寸
        query_layer = query_layer.reshape(
            output_size[2], output_size[0] * output_size[1], -1
        )  # 重塑查询层
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)  # 重塑键层
        matmul_result, qk_time = self._apply_QK(query_layer, key_layer)  # 计算查询和键的矩阵乘法结果及时间
        attention_scores = matmul_result.view(*output_size)  # 重塑注意力分数
        softmax_results, softmax_time = self._apply_Softmax(attention_scores)  # 计算软最大概率及时间
        context_layer, contex_time = self._apply_Contex(softmax_results, value_layer)  # 计算上下文层及时间
        output, attrn_linear_time = self._apply_Linear(context_layer)  # 计算线性输出及时间
        return output, qkv_time, qk_time, softmax_time, contex_time, attrn_linear_time  # 返回所有输出和时间


# class MegatronFlashAtten(torch.nn.Module):  # 定义MegatronFlashAtten类，继承自torch.nn.Module
class SarathiFlashAtten(torch.nn.Module):  # 定义MegatronFlashAtten类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        # super(MegatronFlashAtten, self).__init__()  # 调用父类的初始化方法
        super(SarathiFlashAtten, self).__init__()  # 调用父类的初始化方法
        self.enable_sequence_parallel = args.enable_sequence_parallel  # 获取是否启用序列并行
        self.tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度

        hidden_size = args.hidden_size  # 获取隐藏层大小
        num_attention_heads = args.num_attention_heads  # 获取注意力头数量
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            dtype = torch.float16  # 设置数据类型为float16
        else:
            dtype = torch.float32  # 否则设置为float32
        device = torch.cuda.current_device()  # 获取当前CUDA设备

        self.atten_weight_1 = torch.rand(
            divide((3 * hidden_size), self.tp), hidden_size, device=device
        ).to(dtype)  # 初始化注意力权重为随机值，并根据并行大小分割

        self.hidden_size_per_partition = divide(hidden_size, self.tp)  # 计算每个分区的隐藏层大小
        self.num_attention_heads_per_partition = divide(num_attention_heads, self.tp)  # 计算每个分区的注意力头数量
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)  # 计算每个注意力头的隐藏层大小
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition  # 设置每个分区的查询组数量

        self.atten_linear_weight = torch.rand(
            hidden_size, self.hidden_size_per_partition, device=device
        ).to(dtype)  # 初始化注意力线性权重为随机值

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_attenqkv(self, hideen_states):  # 定义_apply_attenqkv方法，接受隐藏状态作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数

        output = _forward_impl(
            input=hideen_states,  # 输入隐藏状态
            weight=self.atten_weight_1,  # 使用注意力权重1
            bias=None,  # 无偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=False,  # 禁用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算

        new_tensor_shape = output.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )  # 计算新的张量形状
        output = output.view(*new_tensor_shape)  # 重塑输出张量
        (query_layer, key_layer, value_layer) = torch.split(
            output,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )  # 分割输出为查询、键、值层

        return query_layer, key_layer, value_layer  # 返回查询、键、值层

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_flash_atten(self, q, k, v):  # 定义_apply_flash_atten方法，接受查询、键、值作为输入

        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.seqlen_q,
            self.seqlen_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )  # 使用flash attention未填充函数计算输出

        context_layer = rearrange(output, "(b s) ... -> b s ...", b=self.micro_batch)  # 重排输出为上下文层

        return context_layer  # 返回上下文层

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Linear(self, context_layer):  # 定义_apply_Linear方法，接受上下文层作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()  # 重排上下文层
        output_parallel = _forward_impl(
            input=context_layer,  # 输入上下文层
            weight=self.atten_linear_weight,  # 使用注意力线性权重
            bias=None,  # 无偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=False,  # 禁用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算
        return output_parallel  # 返回并行输出

    def forward(self, hidden_state):  # 定义前向传播方法，接受隐藏状态作为输入
        if rearrange is None:  # 如果rearrange函数不可用
            raise ImportError(
                "The function 'rearrange' from 'einops' is required but not available."  # 引发导入错误
            )
        result, qkv_time = self._apply_attenqkv(hidden_state)  # 获取查询、键、值层输出和时间
        q, k, v = result  # 解包查询、键、值层
        q, k, v = [rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)]  # 重排查询、键、值层
        self.micro_batch, self.seqlen_q = q.shape[0], q.shape[1]  # 获取微批处理大小和查询序列长度
        self.seqlen_k = k.shape[1]  # 获取键序列长度

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]  # 重排查询、键、值层
        self.cu_seqlens_q = torch.arange(
            0,
            (self.micro_batch + 1) * self.seqlen_q,
            step=self.seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )  # 生成查询序列长度的CUDA序列

        if self.training:  # 如果是训练模式
            assert self.seqlen_k == self.seqlen_q  # 确保键序列长度等于查询序列长度
            self.cu_seqlens_k = self.cu_seqlens_q  # 设置键序列长度
        context_layer, flash_time = self._apply_flash_atten(q, k, v)  # 计算上下文层和时间
        output, attrn_linear_time = self._apply_Linear(context_layer)  # 计算线性输出和时间
        return output, qkv_time, flash_time, attrn_linear_time  # 返回所有输出和时间



# class MegatronMlp(torch.nn.Module):  # 定义MegatronMlp类，继承自torch.nn.Module
class SarathiMlp(torch.nn.Module):  # 定义MegatronMlp类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        # super(MegatronMlp, self).__init__()  # 调用父类的初始化方法
        super(SarathiMlp, self).__init__()  # 调用父类的初始化方法
        self.tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        self.enable_sequence_parallel = args.enable_sequence_parallel  # 获取是否启用序列并行
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度
        self.add_bias_linear = False  # 初始化是否添加线性偏置为False
        if args.add_bias_linear:  # 如果需要添加线性偏置
            self.add_bias_linear = True  # 设置添加线性偏置为True

        hidden_size = args.hidden_size  # 获取隐藏层大小
        ffn_hidden_size = args.ffn_hidden_size  # 获取前馈网络隐藏层大小
        if args.gated_linear_unit:  # 如果使用门控线性单元
            ffn_hidden_size *= 2  # 前馈网络隐藏层大小乘以2
        num_attention_heads = args.num_attention_heads  # 获取注意力头数量
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            dtype = torch.float16  # 设置数据类型为float16
        else:
            dtype = torch.float32  # 否则设置为float32
        # activation  # 激活函数相关的设置
        if args.openai_gelu:  # 如果使用OpenAI GELU激活函数
            self.activation_func = openai_gelu  # 设置激活函数为OpenAI GELU
        elif args.onnx_safe:  # 如果使用ONNX安全版本的激活函数
            self.activation_func = erf_gelu  # 设置激活函数为误差函数GELU
        elif args.swiglu:  # 如果使用Swiglu激活函数

            def swiglu(x):  # 定义Swiglu激活函数
                x = torch.chunk(x, 2, dim=-1)  # 将输入在最后一个维度上分成两半

                return F.silu(x[0]) * x[1]  # 应用Silu激活并相乘

            self.activation_func = swiglu  # 设置激活函数为Swiglu
        elif args.squared_relu:  # 如果使用平方ReLU激活函数

            def squared_relu(x):  # 定义平方ReLU激活函数
                return torch.pow(F.relu(x), 2)  # 应用ReLU并平方

            self.activation_func = squared_relu  # 设置激活函数为平方ReLU
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion  # 获取是否融合偏置和GELU
            self.activation_func = F.gelu  # 默认设置激活函数为GELU
        output_size_per_partition = divide(ffn_hidden_size, self.tp)  # 计算每个并行分区的输出大小

        self.weight_1 = torch.rand(
            output_size_per_partition, hidden_size, device=device
        ).to(dtype)  # 初始化第一层权重为随机值
        self.bias = torch.empty(output_size_per_partition, device=device).to(dtype)  # 初始化偏置为指定大小的空张量

        self.weight_2 = torch.rand(
            hidden_size, args.ffn_hidden_size // self.tp, device=device
        ).to(dtype)  # 初始化第二层权重为随机值

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Linear1(self, hidden_state):  # 定义_apply_Linear1方法，接受隐藏状态作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数
        output_parallel = _forward_impl(
            input=hidden_state,  # 输入隐藏状态
            weight=self.weight_1,  # 使用第一层权重
            bias=None,  # 不使用偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=False,  # 禁用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算
        return output_parallel  # 返回并行输出

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_activation(self, hidden_state):  # 定义_apply_activation方法，接受隐藏状态作为输入
        if self.add_bias_linear:  # 如果需要添加线性偏置
            intermediate_parallel = self.activation_func(hidden_state + self.bias)  # 应用激活函数并加上偏置
        else:
            intermediate_parallel = self.activation_func(hidden_state)  # 仅应用激活函数

        return intermediate_parallel  # 返回中间输出

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Linear2(self, hidden_state):  # 定义_apply_Linear2方法，接受隐藏状态作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数
        output_parallel = _forward_impl(
            input=hidden_state,  # 输入隐藏状态
            weight=self.weight_2,  # 使用第二层权重
            bias=None,  # 不使用偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=False,  # 禁用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算
        return output_parallel  # 返回并行输出

    def forward(self, hidden_state):  # 定义前向传播方法，接受隐藏状态作为输入
        l1_out, l1_time = self._apply_Linear1(hidden_state)  # 应用第一层线性变换，获取输出及时间
        act_out, act_time = self._apply_activation(l1_out)  # 应用激活函数，获取输出及时间
        l2_out, l2_time = self._apply_Linear2(act_out)  # 应用第二层线性变换，获取输出及时间
        return l2_out, l1_time, act_time, l2_time  # 返回最终输出及各步骤的时间


class logit(torch.nn.Module):  # 定义logit类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        super(logit, self).__init__()  # 调用父类的初始化方法
        self.enable_sequence_parallel = args.enable_sequence_parallel  # 获取是否启用序列并行
        self.tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度
        vocab_size = args.padded_vocab_size  # 获取填充后的词汇表大小
        hidden_size = args.hidden_size  # 获取隐藏层大小
        ffn_hidden_size = args.ffn_hidden_size  # 获取前馈网络隐藏层大小
        if args.gated_linear_unit:  # 如果使用门控线性单元
            ffn_hidden_size *= 2  # 前馈网络隐藏层大小乘以2
        num_attention_heads = args.num_attention_heads  # 获取注意力头数量
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            dtype = torch.float16  # 设置数据类型为float16
        else:
            dtype = torch.float32  # 否则设置为float32
        output_size_per_partition = divide(vocab_size, self.tp)  # 计算每个并行分区的输出大小
        self.word_embeddings_weight = torch.rand(
            output_size_per_partition, hidden_size, device=device, requires_grad=True
        ).to(dtype)  # 初始化词嵌入权重为随机值，并设置需要梯度

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply(self, hidden_state):  # 定义_apply方法，接受隐藏状态作为输入
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce  # 获取线性前向实现函数
        output_parallel = _forward_impl(
            input=hidden_state,  # 输入隐藏状态
            weight=self.word_embeddings_weight,  # 使用词嵌入权重
            bias=None,  # 不使用偏置
            gradient_accumulation_fusion=True,  # 启用梯度累积融合
            async_grad_allreduce=True,  # 启用异步梯度全归约
            sequence_parallel=self.enable_sequence_parallel,  # 设置序列并行
            tp=self.tp,  # 设置并行大小
        )  # 执行前向计算
        return output_parallel  # 返回并行输出

    def forward(self, input):  # 定义前向传播方法，接受输入作为输入
        log_out, log_time = self._apply(input)  # 应用线性变换，获取输出及时间
        return log_out, log_time  # 返回输出及时间


class Grad_param:  # 定义Grad_param类
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        param = args.model_param  # 获取模型参数大小
        self.dp = args.dp_num  # 获取数据并行数

        device = torch.cuda.current_device()  # 获取当前CUDA设备
        dtype = torch.float32  # 设置数据类型为float32
        self.data = torch.rand(param//tp, device=device).to(dtype)  # 初始化数据为随机值，并根据并行大小分割

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply(self):  # 定义_apply方法
        self.data /= self.dp  # 将数据除以数据并行数
        # assert self.data.numel() % self.dp == 0  # 确认数据元素数量可以被数据并行数整除
        shard_size = self.data.numel() // self.dp  # 计算每个分片的大小
        sharded_buffer = [
            self.data[(r * shard_size) : ((r + 1) * shard_size)] for r in range(self.dp)
        ]  # 将数据分割为多个分片
        return sharded_buffer  # 返回分片后的缓冲区


class SequentialMLP(torch.nn.Module):  # 定义SequentialMLP类，继承自torch.nn.Module
    """An implementation of the Experts layer using a sequence of MLP layers.
    
    This class executes each expert sequentially.
    """  # 文档字符串，描述SequentialMLP类的功能

    def __init__(self, num_local_experts, args=None):  # 初始化方法，接受本地专家数量和可选的args参数
        super(SequentialMLP, self).__init__()  # 调用父类的初始化方法
        tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        ep = args.expert_model_parallel_size  # 获取专家模型并行大小
        num_experts = args.num_experts  # 获取专家总数
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度
        topk = args.moe_router_topk  # 获取MoE路由器的topk参数
        hidden_size = args.hidden_size  # 获取隐藏层大小
        num_attention_heads = args.num_attention_heads  # 获取注意力头数量
        self.add_bias = False  # 初始化是否添加偏置为False
        # self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts  # 设置本地专家数量
        self.local_experts = torch.nn.ModuleList()  # 初始化本地专家的模块列表
        for _ in range(self.num_local_experts):  # 遍历本地专家数量
            # expert = MegatronMlp(args)  # 创建一个MegatronMlp实例
            expert = SarathiMlp(args)  # 创建一个MegatronMlp实例
            self.local_experts.append(expert)  # 将专家添加到模块列表中

    def forward(self, permuted_local_hidden_states, tokens_per_expert):  # 定义前向传播方法，接受排列后的本地隐藏状态和每个专家的令牌数量
        output_local = torch.zeros_like(permuted_local_hidden_states)  # 初始化本地输出为与隐藏状态相同形状的零张量
        output_bias_local = None  # 初始化本地偏置输出为None
        if self.add_bias:  # 如果需要添加偏置
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)  # 初始化本地偏置输出为与隐藏状态相同形状的零张量

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)  # 计算每个专家的累计令牌数
        # Insert zero at the beginning for offset index's convenience  # 在开头插入零以方便偏移索引
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)  # 创建一个单元素的零张量
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))  # 将零张量与累计令牌数拼接
        mlp_linear_1_all, mlp_gelu_all, mlp_linear_2_all = 0, 0, 0  # 初始化所有MLP步骤的时间为0
        for expert_num, expert in enumerate(self.local_experts):  # 遍历每个本地专家
            start = cumsum_num_tokens[expert_num]  # 获取当前专家的开始索引
            end = cumsum_num_tokens[expert_num + 1]  # 获取当前专家的结束索引

            hidden = permuted_local_hidden_states[start:end]  # 获取当前专家的隐藏状态
            # output, output_bias = expert(hidden)
            output, mlp_linear_1, mlp_gelu, mlp_linear_2 = expert(hidden)  # 应用专家，获取输出及各步骤时间

            output_local[start:end] = output  # 将输出赋值到本地输出的相应部分
            mlp_linear_1_all += mlp_linear_1  # 累加线性层1的时间
            mlp_gelu_all += mlp_gelu  # 累加GELU层的时间
            mlp_linear_2_all += mlp_linear_2  # 累加线性层2的时间
            if self.add_bias:  # 如果需要添加偏置
                output_bias = output_bias.expand_as(output)  # 扩展偏置以匹配输出形状
                output_bias_local[start:end, :] = output_bias  # 将偏置赋值到本地偏置输出的相应部分

        return output_local, mlp_linear_1_all, mlp_gelu_all, mlp_linear_2_all  # 返回本地输出及各步骤的总时间


class GroupedMLP(torch.nn.Module):  # 定义GroupedMLP类，继承自torch.nn.Module
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.
    
    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """  # 文档字符串，描述GroupedMLP类的功能

    def __init__(self, num_local_experts, args=None):  # 初始化方法，接受本地专家数量和可选的args参数
        super(GroupedMLP, self).__init__()  # 调用父类的初始化方法
        self.num_local_experts = num_local_experts  # 设置本地专家数量
        gg.assert_grouped_gemm_is_available()  # 确认GroupedGEMM是否可用
        tp = args.tensor_model_parallel_size  # 获取张量模型并行大小
        self.hidden_size = args.hidden_size  # 获取隐藏层大小
        self.expert_parallel = args.expert_model_parallel_size > 1  # 确定是否使用专家并行
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            dtype = torch.float16  # 设置数据类型为float16
        else:
            dtype = torch.float32  # 否则设置为float32
        if args.openai_gelu:  # 如果使用OpenAI GELU激活函数
            self.activation_func = openai_gelu  # 设置激活函数为OpenAI GELU
        elif args.onnx_safe:  # 如果使用ONNX安全版本的激活函数
            self.activation_func = erf_gelu  # 设置激活函数为误差函数GELU
        elif args.swiglu:  # 如果使用Swiglu激活函数
            def swiglu(x):  # 定义Swiglu激活函数
                x = torch.chunk(x, 2, dim=-1)  # 将输入在最后一个维度上分成两半

                return F.silu(x[0]) * x[1]  # 应用Silu激活并相乘
            self.activation_func = swiglu  # 设置激活函数为Swiglu
        elif args.squared_relu:  # 如果使用平方ReLU激活函数
            def squared_relu(x):  # 定义平方ReLU激活函数
                return torch.pow(F.relu(x), 2)  # 应用ReLU并平方
            self.activation_func = squared_relu  # 设置激活函数为平方ReLU
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion  # 获取是否融合偏置和GELU
            self.activation_func = F.gelu  # 默认设置激活函数为GELU
        # if args.gated_linear_unit:
        #     def glu(x):
        #         x = torch.chunk(x, 2, dim=-1)
        #         return self.config.activation_func(x[0]) * x[1]
        #     self.activation_func = glu
        # else:
        #     self.activation_func = self.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        # self.moe_extended_tp = config.moe_extended_tp
        # if config.moe_extended_tp:
        #     tp_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # else:
        #     tp_size = parallel_state.get_tensor_model_parallel_world_size()

        fc1_output_size = args.ffn_hidden_size * self.num_local_experts  # 计算fc1的输出大小
        if args.gated_linear_unit:  # 如果使用门控线性单元
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2  # fc1输出大小乘以2
        fc1_output_size_per_partition = divide(fc1_output_size, tp)  # 计算每个并行分区的fc1输出大小

        fc2_input_size = args.ffn_hidden_size * self.num_local_experts  # 计算fc2的输入大小
        fc2_input_size_per_partition = divide(fc2_input_size, tp)  # 计算每个并行分区的fc2输入大小

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.weight1 = torch.rand(self.hidden_size,
                                   fc1_output_size_per_partition,
                                   device=device).to(dtype)  # 初始化第一层权重为随机值
        self.weight2 = torch.rand(fc2_input_size_per_partition,
                                   self.hidden_size,
                                   device=device).to(dtype)  # 初始化第二层权重为随机值

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Linear1(self, permuted_local_hidden_states, tokens_per_expert, w1):  # 定义_apply_Linear1方法，接受排列后的本地隐藏状态、每个专家的令牌数量和权重1作为输入

        fc1_output = gg.ops.gmm(
            permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
        )  # 使用GroupedGEMM进行矩阵乘法
        return fc1_output  # 返回fc1的输出

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_activation(self, fc1_output):  # 定义_apply_activation方法，接受fc1的输出作为输入

        intermediate_parallel = self.activation_func(fc1_output)  # 应用激活函数

        return intermediate_parallel  # 返回中间输出

    @cuda_timing_decorator  # 使用CUDA计时装饰器
    def _apply_Linear2(self, intermediate_parallel, tokens_per_expert, w2):  # 定义_apply_Linear2方法，接受中间输出、每个专家的令牌数量和权重2作为输入

        fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)  # 使用GroupedGEMM进行矩阵乘法

        return fc2_output  # 返回fc2的输出

    def forward(self, permuted_local_hidden_states, tokens_per_expert):  # 定义前向传播方法，接受排列后的本地隐藏状态和每个专家的令牌数量
        w1 = self.weight1.view(self.num_local_experts, self.hidden_size, -1)  # 重塑权重1为适合GroupedGEMM的形状
        w2 = self.weight2.view(self.num_local_experts, -1, self.hidden_size)  # 重塑权重2为适合GroupedGEMM的形状
        l1_out, l1_time = self._apply_Linear1(permuted_local_hidden_states, tokens_per_expert, w1)  # 应用第一层线性变换，获取输出及时间
        act_out, act_time = self._apply_activation(l1_out)  # 应用激活函数，获取输出及时间
        l2_out, l2_time = self._apply_Linear2(act_out, tokens_per_expert, w2)  # 应用第二层线性变换，获取输出及时间

        return l2_out, l1_time, act_time, l2_time  # 返回最终输出及各步骤的时间


class MoELayer(torch.nn.Module):  # 定义MoELayer类，继承自torch.nn.Module
    def __init__(self, args=None):  # 初始化方法，接受可选的args参数
        super(MoELayer, self).__init__()  # 调用父类的初始化方法

        ep = args.expert_model_parallel_size  # 获取专家模型并行大小
        num_experts = args.num_experts  # 获取专家总数
        micro_batch = args.micro_batch  # 获取微批处理大小
        seq_len = args.seq_length  # 获取序列长度
        topk = args.moe_router_topk  # 获取MoE路由器的topk参数
        hidden_size = args.hidden_size  # 获取隐藏层大小
        num_attention_heads = args.num_attention_heads  # 获取注意力头数量
        self.num_local_experts = int(num_experts / ep)  # 计算每个并行分区的本地专家数量
        if args.dtype == "bfloat16":  # 如果数据类型为bfloat16
            dtype = torch.bfloat16  # 设置数据类型为bfloat16
        elif args.dtype == "float16":  # 如果数据类型为float16
            dtype = torch.float16  # 设置数据类型为float16
        else:
            dtype = torch.float32  # 否则设置为float32
        device = torch.cuda.current_device()  # 获取当前CUDA设备
        if args.moe_grouped_gemm:  # 如果使用MoE的GroupedGEMM
            self.experts = GroupedMLP(self.num_local_experts, args)  # 初始化GroupedMLP专家
        else:
            self.experts = SequentialMLP(self.num_local_experts, args)  # 初始化SequentialMLP专家
        # print("aa", seq_len * micro_batch * topk * dp / num_experts * self.num_local_experts)
        self.dispatched_input = torch.rand(int(seq_len * micro_batch * topk * ep / num_experts * self.num_local_experts), hidden_size,
                                           device=device).to(dtype)  # 初始化分发的输入为随机值
        temp_val = int(seq_len * micro_batch * topk * ep / num_experts)  # 计算每个专家的令牌数量
        # self.tokens_per_expert = torch.tensor([temp,temp], device=device)

        self.tokens_per_expert = torch.full((self.num_local_experts,), temp_val)  # 创建一个全为temp_val的张量，表示每个本地专家的令牌数量
        # print('aa', self.tokens_per_expert)

    def forward(self, hidden_states: torch.Tensor):  # 定义前向传播方法，接受隐藏状态作为输入
        # probs, indices = self.router(hidden_states)
        # (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
        #     hidden_states, probs, indices
        # )
        
        expert_output, mlp_linear_1, mlp_gelu, mlp_linear_2 = self.experts(self.dispatched_input, self.tokens_per_expert)  # 应用专家，获取输出及各步骤时间
        # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        return expert_output, mlp_linear_1, mlp_gelu, mlp_linear_2  # 返回专家输出及各步骤的时间

