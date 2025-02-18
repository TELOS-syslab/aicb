from dataclasses import asdict  # 导入 asdict 函数，用于将数据类实例转换为字典
from typing import Any, Dict, Optional  # 导入类型注解工具

import torch  # 导入 PyTorch 库，用于张量操作和深度学习
from sarathi.config import ParallelConfig  # 导入 ParallelConfig 类，用于并行配置

from vidur.config.model_config import BaseModelConfig  # 导入 BaseModelConfig 类，用于模型基础配置
from vidur.types import ActivationType, NormType  # 导入激活函数类型和归一化类型


class ModelConfig:
    def __init__(
        self,
        name: str,  # 模型名称
        num_layers: int,  # 模型层数
        num_q_heads: int,  # 查询头数量
        num_kv_heads: int,  # 键值头数量
        embedding_dim: int,  # 嵌入维度
        mlp_hidden_dim: int,  # MLP 隐藏层维度
        max_position_embeddings: int,  # 最大位置嵌入数
        use_gated_mlp: bool,  # 是否使用门控 MLP
        use_bias: bool,  # 是否使用偏置
        use_qkv_bias: bool,  # 是否在 QKV 计算中使用偏置
        activation: ActivationType,  # 激活函数类型
        norm: NormType,  # 归一化类型
        post_attn_norm: bool,  # 是否在注意力后使用归一化
        vocab_size: int,  # 词汇表大小
        is_neox_style: Optional[bool] = True,  # 是否使用 NeoX 风格的旋转位置编码
        rope_theta: Optional[int] = None,  # RoPE 的 theta 参数
        rope_scaling: Optional[Dict[str, Any]] = None,  # RoPE 缩放参数
        partial_rotary_factor: float = 1.0,  # 部分旋转因子
        no_tensor_parallel: bool = False,  # 是否禁用张量并行
    ):
        self.name = name  # 初始化模型名称
        self.num_layers = num_layers  # 初始化模型层数
        self.num_q_heads = num_q_heads  # 初始化查询头数量
        self.num_kv_heads = num_kv_heads  # 初始化键值头数量
        self.embedding_dim = embedding_dim  # 初始化嵌入维度
        self.mlp_hidden_dim = mlp_hidden_dim  # 初始化 MLP 隐藏层维度
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入数
        self.use_gated_mlp = use_gated_mlp  # 初始化是否使用门控 MLP
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.use_bias = use_bias  # 初始化是否使用偏置
        self.use_qkv_bias = use_qkv_bias  # 初始化是否在 QKV 计算中使用偏置
        self.activation = str(activation)  # 将激活函数类型转换为字符串并初始化
        self.norm = str(norm)  # 将归一化类型转换为字符串并初始化
        self.post_attn_norm = post_attn_norm  # 初始化是否在注意力后使用归一化
        self.no_tensor_parallel = no_tensor_parallel  # 初始化是否禁用张量并行
        self.partial_rotary_factor = partial_rotary_factor  # 初始化部分旋转因子
        self.rope_theta = rope_theta  # 初始化 RoPE 的 theta 参数
        self.rope_scaling = rope_scaling  # 初始化 RoPE 缩放参数
        self.is_neox_style = is_neox_style  # 初始化是否使用 NeoX 风格的旋转位置编码

        assert self.norm in ["layer_norm", "rms_norm"]  # 确保归一化类型是 layer_norm 或 rms_norm
        assert self.activation in ["gelu", "silu"]  # 确保激活函数类型是 gelu 或 silu

        if self.use_gated_mlp:  # 如果使用门控 MLP
            assert self.activation == "silu"  # 确保激活函数是 silu
        else:  # 如果不使用门控 MLP
            assert self.activation == "gelu"  # 确保激活函数是 gelu

    @staticmethod
    def from_model_name(model_name: str):  # 根据模型名称创建 ModelConfig 实例的静态方法
        model_config: BaseModelConfig = BaseModelConfig.create_from_name(model_name)  # 根据模型名称创建基础配置
        model_config_dict = asdict(model_config)  # 将基础配置转换为字典

        return ModelConfig(model_name, **model_config_dict)  # 返回 ModelConfig 实例

    def get_num_q_heads(self, parallel_config: ParallelConfig):  # 获取并行配置下的查询头数量
        return self.num_q_heads // parallel_config.tensor_parallel_size  # 计算并返回查询头数量

    def get_num_kv_heads(self, parallel_config: ParallelConfig):  # 获取并行配置下的键值头数量
        return self.num_kv_heads // parallel_config.tensor_parallel_size  # 计算并返回键值头数量

    def get_head_size(self):  # 获取每个头的大小
        return self.embedding_dim // self.num_q_heads  # 计算并返回每个头的大小

    @property
    def dtype(self):  # 定义数据类型的属性
        return torch.float16  # 返回数据类型为 float16
