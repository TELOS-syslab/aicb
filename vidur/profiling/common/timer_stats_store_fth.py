import numpy as np

from vidur.profiling.utils import ProfileMethod
from vidur.profiling.utils.singleton import Singleton


class TimerStatsStore(metaclass=Singleton):
    def __init__(self, profile_method: str, disabled: bool = False):
        self.disabled = disabled  # 是否禁用计时统计
        self.profile_method = ProfileMethod[profile_method.upper()]  # 设定要使用的配置方法
        self.TIMING_STATS = {}  # 初始化计时统计字典

    def record_time(self, name: str, time):
        name = name.replace("vidur_", "")  # 去掉名称中的前缀“vidur_”
        if name not in self.TIMING_STATS:
            self.TIMING_STATS[name] = []  # 如果名称不存在于字典中，初始化该名称的列表
        self.TIMING_STATS[name].append(time)  # 记录时间

    def clear_stats(self):
        self.TIMING_STATS = {}  # 清空计时统计字典

    def get_stats(self):
        stats = {}  # 初始化统计结果字典
        for name, times in self.TIMING_STATS.items():
            times = [
                (time if isinstance(time, float) else time[0].elapsed_time(time[1]))  # 如果时间是float类型直接使用，否则计算经过的时间
                for time in times
            ]

            stats[name] = {
                "min": np.min(times),  # 最小值
                "max": np.max(times),  # 最大值
                "mean": np.mean(times),  # 平均值
                "median": np.median(times),  # 中位数
                "std": np.std(times),  # 标准差
            }

        return stats  # 返回统计结果