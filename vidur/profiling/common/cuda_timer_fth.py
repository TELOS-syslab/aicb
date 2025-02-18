import time  # 导入时间模块

import torch  # 导入 PyTorch 模块
from torch.profiler import record_function  # 从 PyTorch 中导入 record_function 用于性能分析

from vidur.profiling.common.timer_stats_store import TimerStatsStore  # 从 vidur.profiling.common 导入 TimerStatsStore 类
from vidur.profiling.utils import ProfileMethod  # 从 vidur.profiling.utils 导入 ProfileMethod 类

class CudaTimer:  # 定义 CudaTimer 类
    def __init__(  # 初始化方法
        self,  # self 引用
        name,  # 计时器名称
        layer_id: int = 0,  # 我们不关心 layer_id，它只是为了与 sarathi cudatimer 兼容
        aggregation_fn=sum,  # 聚合函数默认为 sum
        filter_str=None,  # 过滤字符串
    ):
        if name:  # 如果名称存在
            # beautify the names we get from vllm
            name = str(name).replace("OperationMetrics.", "")  # 替换名称中的 "OperationMetrics."
            name = name.lower()  # 转换为小写
            self.name = f"vidur_{name}"  # 生成计时器名称
        else:
            self.name = None  # 名称为空

        self.timer_stats_store = TimerStatsStore()  # 创建 TimerStatsStore 实例
        self.disabled = (name is None) or self.timer_stats_store.disabled  # 判断是否禁用

        if self.disabled:  # 如果禁用
            return  # 直接返回

        self.aggregation_fn = aggregation_fn  # 设置聚合函数
        self.filter_str = filter_str  # 设置过滤字符串

        if self.timer_stats_store.profile_method == ProfileMethod.KINETO:  # 如果性能分析方法为 KINETO
            self.profiler = torch.profiler.profile(  # 创建 profiler 实例
                activities=[torch.profiler.ProfilerActivity.CUDA],  # 监视 CUDA 活动
                on_trace_ready=self.handle_trace,  # 当追踪就绪时调用 handle_trace 方法
            )
        else:
            self.profiler = None  # 否则 profiler 为空
        self.start_event = None  # 初始化起始事件为 None
        self.end_event = None  # 初始化结束事件为 None
        self.start_time = None  # 初始化起始时间为 None
        self.end_time = None  # 初始化结束时间为 None

    def __enter__(self):  # 上下文管理器 enter 方法
        if self.disabled:  # 如果禁用
            return  # 直接返回

        if self.timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:  # 如果性能分析方法为 RECORD_FUNCTION
            self.profiler_function_context = record_function(self.name)  # 创建 record_function 上下文
            self.profiler_function_context.__enter__()  # 进入上下文
        elif self.timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:  # 如果性能分析方法为 CUDA_EVENT
            self.start_event = torch.cuda.Event(enable_timing=True)  # 创建启用计时的 CUDA 事件
            self.start_event.record()  # 记录起始事件
        elif self.timer_stats_store.profile_method == ProfileMethod.KINETO:  # 如果性能分析方法为 KINETO
            self.profiler.__enter__()  # 进入 profiler 上下文
        elif self.timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:  # 如果性能分析方法为 PERF_COUNTER
            torch.cuda.synchronize()  # 同步 CUDA
            self.start_time = time.perf_counter()  # 记录起始时间
        else:
            raise ValueError(  # 抛出 ValueError 异常
                f"Unknown profile method {self.timer_stats_store.profile_method}"  # 未知的性能分析方法
            )
        return self  # 返回 self

    def handle_trace(self, trace):  # 处理追踪数据的方法
        events = trace.events()  # 获取追踪的事件

        if self.filter_str:  # 如果过滤字符串存在
            events = [e for e in events if e.name.startswith(self.filter_str)]  # 过滤事件

        total_cuda_time = self.aggregation_fn([e.cuda_time_total for e in events])  # 计算总的 CUDA 时间
        self.timer_stats_store.record_time(  # 记录时间
            self.name, total_cuda_time * 1e-3  # 转换为毫秒
        )  # convert to ms

    def __exit__(self, *args):  # 上下文管理器 exit 方法
        if self.disabled:  # 如果禁用
            return  # 直接返回

        if self.timer_stats_store.profile_method == ProfileMethod.RECORD_FUNCTION:  # 如果性能分析方法为 RECORD_FUNCTION
            self.profiler_function_context.__exit__(*args)  # 退出上下文
        elif self.timer_stats_store.profile_method == ProfileMethod.CUDA_EVENT:  # 如果性能分析方法为 CUDA_EVENT
            self.end_event = torch.cuda.Event(enable_timing=True)  # 创建启用计时的结束事件
            self.end_event.record()  # 记录结束事件
            self.timer_stats_store.record_time(  # 记录时间
                self.name, [self.start_event, self.end_event]  # 传入起始事件和结束事件
            )
        elif self.timer_stats_store.profile_method == ProfileMethod.KINETO:  # 如果性能分析方法为 KINETO
            self.profiler.__exit__(*args)  # 退出 profiler 上下文
        elif self.timer_stats_store.profile_method == ProfileMethod.PERF_COUNTER:  # 如果性能分析方法为 PERF_COUNTER
            torch.cuda.synchronize()  # 同步 CUDA
            self.end_time = time.perf_counter()  # 记录结束时间
            self.timer_stats_store.record_time(  # 记录时间
                self.name, (self.end_time - self.start_time) * 1e3  # 转换为毫秒
            )  # convert to ms
        else:
            raise ValueError(  # 抛出 ValueError 异常
                f"Unknown profile method {self.timer_stats_store.profile_method}"  # 未知的性能分析方法
            )