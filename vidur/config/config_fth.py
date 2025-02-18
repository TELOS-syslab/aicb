import json  # 导入json模块
import os  # 导入操作系统接口模块
from abc import ABC  # 导入抽象基类模块
from dataclasses import dataclass, field  # 导入数据类模块
from datetime import datetime  # 导入日期时间模块
from typing import List, Optional  # 导入类型检查模块

# 导入自定义模块中的基类配置
from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.device_sku_config import BaseDeviceSKUConfig
from vidur.config.flat_dataclass import create_flat_dataclass
from vidur.config.model_config import BaseModelConfig
from vidur.config.node_sku_config import BaseNodeSKUConfig
from vidur.config.utils import dataclass_to_dict
from vidur.logger import init_logger  # 导入日志初始化模块
from vidur.types import (  # 导入自定义类型
    ExecutionTimePredictorType,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)  # 初始化日志记录器

@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):  # 基础请求间隔生成器配置类
    seed: int = field(  # 随机数生成器的种子
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):  # 基础请求长度生成器配置类
    seed: int = field(  # 随机数生成器的种子
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(  # 最大令牌数
        default=4096,
        metadata={"help": "Maximum tokens."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):  # 跟踪请求间隔生成器配置类
    trace_file: str = field(  # 跟踪请求间隔生成器文件的路径
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(  # 跟踪请求间隔生成器的开始时间
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(  # 跟踪请求间隔生成器的结束时间
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(  # 跟踪请求间隔生成器的时间缩放因子
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():  # 获取请求间隔生成器类型的方法
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):  # 泊松请求间隔生成器配置类
    qps: float = field(  # 每秒查询数
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():  # 获取请求间隔生成器类型的方法
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):  # 伽马请求间隔生成器配置类
    qps: float = field(  # 每秒查询数
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(  # 变异系数
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():  # 获取请求间隔生成器类型的方法
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):  # 静态请求间隔生成器配置类
    @staticmethod
    def get_type():  # 获取请求间隔生成器类型的方法
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):  # 跟踪请求长度生成器配置类
    trace_file: str = field(  # 跟踪请求长度生成器文件的路径
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(  # 预填充缩放因子
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(  # 解码缩放因子
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():  # 获取请求长度生成器类型的方法
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):  # 齐夫请求长度生成器配置类
    theta: float = field(  # 齐夫请求长度生成器的theta值
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(  # 是否对齐夫请求长度生成器结果进行打乱
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(  # 最小令牌数
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(  # 预填充和解码比例
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():  # 获取请求长度生成器类型的方法
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):  # 均匀请求长度生成器配置类
    min_tokens: int = field(  # 最小令牌数
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(  # 预填充和解码比例
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():  # 获取请求长度生成器类型的方法
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):  # 固定请求长度生成器配置类
    prefill_tokens: int = field(  # 预填充令牌数
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(  # 解码令牌数
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    @staticmethod
    def get_type():  # 获取请求长度生成器类型的方法
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):  # 基础请求生成器配置类
    seed: int = field(  # 随机数生成器的种子
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):  # 合成请求生成器配置类
    length_generator_config: BaseRequestLengthGeneratorConfig = field(  # 长度生成器配置
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(  # 间隔生成器配置
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(  # 请求数量
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(  # 合成请求生成器的持续时间
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )

    def __post_init__(self):  # 初始化后设置最大令牌数
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():  # 获取请求生成器类型的方法
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):  # 跟踪请求生成器配置类
    trace_file: str = field(  # 跟踪请求生成器文件的路径
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(  # 预填充缩放因子
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(  # 解码缩放因子
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(  # 时间缩放因子
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: int = field(  # 最大令牌数
        default=4096,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():  # 获取请求生成器类型的方法
        return RequestGeneratorType.TRACE_REPLAY


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):  # 基础副本调度器配置类
    batch_size_cap: int = field(  # 最大批量大小上限
        default=128,
        metadata={"help": "Maximum batch size cap."},
    )
    block_size: int = field(  # 块大小
        default=16,
        metadata={"help": "Block size."},
    )
    watermark_blocks_fraction: float = field(  # 水印块占比
        default=0.01,
        metadata={"help": "Watermark blocks fraction."},
    )
    num_blocks: Optional[int] = field(  # 块数量
        default=None,
        metadata={"help": "Number of blocks."},
    )


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):  # VLLM调度器配置类
    max_tokens_in_batch: int = field(  # 每批次最大令牌数
        default=4096,
        metadata={"help": "Maximum tokens in batch for vLLM."},
    )

    @staticmethod
    def get_type():  # 获取副本调度器类型的方法
        return ReplicaSchedulerType.VLLM


@dataclass
class LightllmSchedulerConfig(BaseReplicaSchedulerConfig):  # LightLLM调度器配置类
    max_tokens_in_batch: int = field(  # 每批次最大令牌数
        default=4096,
        metadata={"help": "Maximum tokens in batch for LightLLM."},
    )
    max_waiting_iters: int = field(  # 最大等待迭代次数
        default=10,
        metadata={"help": "Maximum waiting iterations for LightLLM."},
    )

    @staticmethod
    def get_type():  # 获取副本调度器类型的方法
        return ReplicaSchedulerType.LIGHTLLM

@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA  # 返回调度器类型为ORCA


@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER  # 返回调度器类型为FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for Sarathi."},  # Sarathi的块大小
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI  # 返回调度器类型为SARATHI


@dataclass
class MetricsConfig:
    """Metric configuration."""  # 度量配置

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},  # 是否写入度量数据
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},  # 是否写入JSON跟踪
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},  # Weights & Biases项目名称
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},  # Weights & Biases组名称
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},  # Weights & Biases运行名称
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},  # Weights & Biases扫描ID
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},  # Weights & Biases运行ID
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},  # 开启Chrome跟踪
    )
    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},  # 是否保存表格到wandb
    )
    store_plots: bool = field(
        default=True,
        metadata={"help": "Whether to store plots."},  # 是否保存图表
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},  # 是否保存操作度量
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},  # 是否保存token完成度量
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},  # 是否保存请求度量
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},  # 是否保存批处理度量
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},  # 是否保存利用率度量
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to keep individual batch metrics."},  # 是否保留单个批次度量
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},  # 子样本
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},  # 最小批次索引
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},  # 最大批次索引
    )
    output_dir: str = field(
        default="simulator_output",
        metadata={"help": "Output directory."},  # 输出目录
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},  # 缓存目录
    )

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )  # 设置输出目录为带有当前时间戳的格式
        os.makedirs(self.output_dir, exist_ok=True)  # 创建输出目录，如果已存在则忽略


@dataclass
class ReplicaConfig:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Model name."},  # 模型名称
    )
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},  # 内存边距系数
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages."},  # 管道阶段的数量
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},  # 张量并行大小
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},  # 设备
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},  # 网络设备
    )

    def __post_init__(self):
        self.world_size = self.num_pipeline_stages * self.tensor_parallel_size  # 计算世界大小
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )  # 从模型名称生成模型配置
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )  # 从设备类型生成设备配置
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )  # 从网络设备类型生成节点配置


@dataclass
class BaseGlobalSchedulerConfig(BasePolyConfig):
    pass  # 基础全局调度器配置类


@dataclass
class RandomGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM  # 返回全局调度器类型为随机


@dataclass
class RoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.ROUND_ROBIN  # 返回全局调度器类型为轮询


@dataclass
class LORGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOR  # 返回全局调度器类型为LOR


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    compute_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv",
        metadata={"help": "Path to the compute input file."},  # 计算输入文件的路径
    )
    attention_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},  # 注意力输入文件的路径
    )
    all_reduce_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},  # 全缩减输入文件的路径
    )
    send_recv_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},  # 发送接收输入文件的路径
    )
    cpu_overhead_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},  # CPU开销输入文件的路径
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},  # K折交叉验证的分割数
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache prediction models."},  # 是否缓存预测模型
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},  # KV缓存预测粒度
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},  # 最大预填充块大小
    )
    prediction_max_batch_size: int = field(
        default=128,
        metadata={"help": "Max batch size for prediction."},  # 最大批处理大小
    )
    prediction_max_tokens_per_request: int = field(
        default=4096,
        metadata={"help": "Max tokens per request for prediction."},  # 每个请求的最大token数
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention decode batching overhead fraction."},  # 解码批处理的注意力开销系数
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention prefill batching overhead fraction."},  # 预填充批处理的注意力开销系数
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.02,
        metadata={"help": "NCCL CPU launch overhead in ms."},  # NCCL CPU启动开销（毫秒）
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},  # 每个设备的NCCL CPU偏差开销（毫秒）
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},  # 训练作业线程数
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},  # 是否跳过CPU开销建模
    )


@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},  # 线性回归的多项式程度
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},  # 线性回归的多项式是否包含偏差
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},  # 线性回归的多项式是否仅交互
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},  # 线性回归是否拟合截距
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION  # 返回执行时间预测器类型为线性回归


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},  # 随机森林的估计器数
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},  # 随机森林的最大深度
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},  # 随机森林的最小样本分割数
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST  # 返回执行时间预测器类型为随机森林


@dataclass
class ClusterConfig:
    num_replicas: int = field(
        default=1,
        metadata={"help": "Number of replicas."},  # 副本数
    )
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    global_scheduler_config: BaseGlobalSchedulerConfig = field(
        default_factory=RoundRobinGlobalSchedulerConfig,
        metadata={"help": "Global scheduler config."},  # 全局调度器配置
    )
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig,
        metadata={"help": "Replica scheduler config."},  # 副本调度器配置
    )

# 在下面的代码中：
# 冒号（:）：
# 在 Python 中，冒号用于定义数据类型和默认值。在使用 dataclass 时，冒号用于指定类属性的类型。例如，seed: int 表示 seed 是一个整数类型的属性。
# 这种用法主要有助于类型检查和代码的自我文档化，使得代码更易于理解和维护。
# field：
# field 是 dataclasses 模块中的一个函数，用于指定数据类字段的默认值、默认工厂函数以及其他元数据信息。
# 在代码中，例如 seed: int = field(default=42, ...)，field 函数用于为 seed 属性提供一个默认值 42，以及通过 metadata 提供一些附加信息（如帮助文档）。
# default_factory 是 field 的一个可选参数，用于在初始化时调用工厂函数来生成默认值。
# 总体来说，使用冒号和 field 能够帮助定义类的属性及其默认行为，并能更好地管理类的初始化和自动生成代码。

# @dataclass 是 Python 中的一个装饰器，位于 dataclasses 模块中，用于简化类的创建。它可以自动为类生成一些常用的特殊方法，如 __init__()、__repr__()、__eq__() 等，从而减少样板代码，提高代码的可读性和维护性。
@dataclass
class SimulationConfig(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},  # 随机数生成器的种子
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},  # 日志记录级别
    )
    time_limit: int = field(
        default=0,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},  # 模拟的时间限制，以秒为单位，0表示无限制
    )
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster config."},  # 集群配置
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "Request generator config."},  # 请求生成器配置
    )
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},  # 执行时间预测器配置
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics config."},  # 度量配置
    )

    def __post_init__(self):
        self.write_config_to_file()  # 初始化完成后，将配置写入文件

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()  # 从命令行参数创建扁平配置
        instance = flat_config.reconstruct_original_dataclass()  # 重建原始数据类
        instance.__flat_config__ = flat_config  # 存储扁平配置
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):  # 如果没有扁平配置
            logger.warning("Flat config not found. Returning the original config.")  # 记录警告，返回原始配置
            return self.__dict__

        return self.__flat_config__.__dict__  # 返回扁平配置

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)  # 将数据类转换为字典
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)  # 将配置字典保存为JSON文件