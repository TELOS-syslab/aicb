"""
Copyright (c) 2021, Alibaba Group;  # 版权声明，标明代码归属阿里巴巴集团
Licensed under the Apache License, Version 2.0 (the "License");  # 声明代码遵循Apache License 2.0版本许可
you may not use this file except in compliance with the License.  # 除非符合许可协议，否则不得使用此文件
You may obtain a copy of the License at  # 可以在以下地址获取许可的副本
   http://www.apache.org/licenses/LICENSE-2.0  # Apache许可证的官方链接
Unless required by applicable law or agreed to in writing, software  # 除非法律要求或书面同意
distributed under the License is distributed on an "AS IS" BASIS,  # 根据许可分发的软件是“按原样”提供的
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  # 不提供任何形式的明示或暗示保证
See the License for the specific language governing permissions and  # 查看许可以了解特定的权限和限制
limitations under the License.  # 许可的限制条款
"""

import os, math  # 导入os和math模块，用于文件操作和数学计算
import pickle  # 导入pickle模块，用于序列化和反序列化对象
import csv  # 导入csv模块，用于处理CSV文件
import dataclasses  # 导入dataclasses模块，用于定义数据类
import numpy as np  # 导入numpy库，用于科学计算
from typing import Union, Dict, List  # 导入typing模块中的类型提示工具
from utils.utils import CommType, CommGroup  # 导入通信类型和通信组相关的工具类
from log_analyzer.utils import convert_size_to_msg, calc_bw_log  # 导入日志分析器中的工具函数
import copy  # 导入copy模块，用于深拷贝对象

@dataclasses.dataclass  # 使用dataclass装饰器定义数据类
class LogItem:  # 定义一个日志项类
    comm_type: CommType = dataclasses.field(default=None)  # 通信类型，默认为None
    comm_group: CommGroup = dataclasses.field(default=None)  # 通信组，默认为None
    comm_group_size: int = dataclasses.field(default=None)  # 通信组大小，默认为None
    msg_size: float = dataclasses.field(default=0)  # 消息大小，默认为0

    stage: str = dataclasses.field(default="")  # 阶段名称，默认为空字符串
    dst: int = dataclasses.field(default=None)  # 目标节点，默认为None
    src: int = dataclasses.field(default=None)  # 源节点，默认为None
    additional: str = dataclasses.field(default="")  # 附加信息，默认为空字符串

    _elapsed_time: float = dataclasses.field(default=None)  # 耗时，默认为None
    algbw: float = dataclasses.field(default=None)  # 算法带宽，默认为None
    busbw: float = dataclasses.field(default=None)  # 总线带宽，默认为None
    count: float = dataclasses.field(default=1)  # 计数，默认为1

    @property  # 定义属性方法，用于获取耗时
    def elapsed_time(self) -> float:
        return self._elapsed_time  # 返回耗时

    @elapsed_time.setter  # 定义setter方法，用于设置耗时
    def elapsed_time(self, elapsed_time):
        self._elapsed_time = elapsed_time  # 设置耗时
        self.algbw, self.busbw = calc_bw_log(  # 计算算法带宽和总线带宽
            self.comm_type, self.msg_size, elapsed_time, self.comm_group_size
        )

    def is_epoch_end(self):  # 判断是否为epoch结束
        return self.comm_type == CommType.epoch_end  # 如果通信类型为epoch_end，则返回True

    def is_workload(self):  # 判断是否为工作负载
        return self.elapsed_time is None  # 如果耗时为None，则返回True

    def view_as_ds_log(self):  # 将日志项格式化为字符串
        log_str = f"[RANK 0] comm op: {self.comm_type} | comm group: {self.comm_group}"  # 构建日志字符串
        log_str += " | time (ms): {:.2f}".format(self.elapsed_time)  # 添加耗时信息
        if self.comm_type == CommType.computation or self.additional == 'overlap':  # 如果是计算或重叠操作
            log_str += " | msg size: " + '0'  # 消息大小为0
            log_str += " | algbw (GB): " + '0'  # 算法带宽为0
            log_str += " | busbw (GB): " + '0'  # 总线带宽为0
        else:  # 否则
            log_str += " | msg size: " + convert_size_to_msg(self.msg_size)  # 添加消息大小
            log_str += " | algbw (GB): {:.2f} ".format(self.algbw)  # 添加算法带宽
            log_str += " | busbw (GB): {:.2f} ".format(self.busbw)  # 添加总线带宽
        return log_str  # 返回格式化后的日志字符串

    def csv_header(self):  # 获取CSV文件的表头
        return ",".join([k for k in self.__dict__.keys()])  # 将所有字段名用逗号连接成字符串

    def view_as_csv_line(self):  # 将日志项格式化为CSV行
        return ",".join([str(getattr(self, k)) for k in self.__dict__.keys()])  # 将所有字段值用逗号连接成字符串

    def __str__(self):  # 定义字符串表示方法
        if self.is_workload():  # 如果是工作负载
            return "None"  # 返回"None"
        return "None"  # 否则也返回"None"


def _print_stage_log(stage_name: str, stage_count: int, comm_type_info: Dict, primary_key: List[str], agg_key: List[str], performance_key: List[str], busbw_key: List[str]):  # 打印阶段日志
    header = f"{'Comm_Type':<15} {'Comm_Group':<12} {'Message_Size':<12} {'Count':<12} {'Avg_Elapsed_Time ± Std ':<24} {'Avg_BusBw ± Std':<24}\n"  # 定义表头
    separator = "-" * len(header) + "\n"  # 定义分隔符
    log_str = separator + header + separator  # 构建日志字符串

    for pkey in sorted(comm_type_info.keys()):  # 遍历通信类型信息
        row_str = ""  # 初始化行字符串
        values = {}  # 初始化值字典
        for i, pkey_name in enumerate(primary_key):  # 遍历主键
            value = pkey[i] if pkey_name != "msg_size" else convert_size_to_msg(pkey[i])  # 获取值并转换消息大小
            values[pkey_name] = value  # 存储值
        for key in agg_key:  # 遍历聚合键
            value = comm_type_info[pkey][key]  # 获取值
            value = convert_size_to_msg(value) if key == "msg_size" else f"{value:.2f}"  # 转换消息大小或格式化数值
            values[key] = value  # 存储值
        for key in performance_key:  # 遍历性能键
            performance_value_list = sorted(comm_type_info[pkey][key])  # 获取并排序性能值列表
            values[f'avg_{key}'] = f"{np.mean(performance_value_list):.2f}±{np.std(performance_value_list):.2f}"  # 计算平均值和标准差
            values[f'min_{key}'] = f"{performance_value_list[0]:.2f}"  # 获取最小值
            values[f'max_{key}'] = f"{performance_value_list[-1]:.2f}"  # 获取最大值
        
        for key in busbw_key:  # 遍历总线带宽键
            busbw_value_list = sorted(comm_type_info[pkey][key])  # 获取并排序总线带宽值列表
            values[f'avg_{key}'] = f"{np.mean(busbw_value_list):.2f}±{np.std(busbw_value_list):.2f}"  # 计算平均值和标准差

        row_str += f"{values['comm_type']:<15} {values['comm_group']:<12} {values['msg_size']:<12} {values['count']:<16} {values['avg__elapsed_time']:<24} {values['avg_busbw']:<18}\n"  # 构建行字符串
        log_str += row_str  # 添加到日志字符串

    return log_str  # 返回日志字符串


def _analyze_stage_log(comm_log: List[Dict], stage: str, comm_info: Dict[str, Dict]):  # 分析阶段日志
    def __update_info(  # 更新信息的内部函数
        info_dict,
        log,
        primary_key: List[str],
        agg_key: List[str],
        performance_key: List[str],
        busbw_key: List[str],
    ):
        primary_key = tuple(log[key] for key in primary_key)  # 获取主键
        if primary_key not in info_dict:  # 如果主键不在信息字典中
            info_dict[primary_key] = dict((key, 0) for key in agg_key)  # 初始化聚合键
            info_dict[primary_key].update(dict((key, []) for key in performance_key))  # 初始化性能键
            info_dict[primary_key].update(dict((key, []) for key in busbw_key))  # 初始化总线带宽键
        for key in agg_key:  # 遍历聚合键
            info_dict[primary_key][key] += log[key]  # 累加值
        for key in performance_key:  # 遍历性能键
            info_dict[primary_key][key].append(log[key])  # 添加值
        for key in busbw_key:  # 遍历总线带宽键
            info_dict[primary_key][key].append(log[key])  # 添加值

    if stage not in comm_info:  # 如果阶段不在通信信息中
        comm_info[stage] = {  # 初始化阶段信息
            "count": 0,  # 计数
            "comm_type_info": {},  # 通信类型信息
            "detailed_comm_type_info": {},  # 详细通信类型信息
        }
    comm_info[stage]["count"] += 1  # 增加计数
    # key: comm_type, value: count, time_ms
    comm_type_info = comm_info[stage]["comm_type_info"]  # 获取通信类型信息
    # key: comm_type, msg_size, value: count, time_ms
    detailed_comm_type_info = comm_info[stage]["detailed_comm_type_info"]  # 获取详细通信类型信息
    for log in comm_log:  # 遍历通信日志
        if log.comm_type != CommType.computation:  # 如果通信类型不是计算
            __update_info(  # 更新信息
                comm_type_info,
                log.__dict__,
                ["comm_type", "comm_group"],
                ["count", "msg_size"],
                ["_elapsed_time"],
                ["busbw"],
            )
            __update_info(  # 更新详细信息
                detailed_comm_type_info,
                log.__dict__,
                ["comm_type", "comm_group", "msg_size"],
                ["count"],
                ["_elapsed_time"],
                ["busbw"],
            )


class Log:  # 定义日志类
    def __init__(self) -> None:  # 初始化方法
        self.comm_logs = []  # 初始化通信日志列表
        self.comm_log_each_epoch = [[]]  # 初始化每个epoch的通信日志列表
        self.epoch_times = []  # 初始化epoch时间列表

    def add_comm_log(self, comm_log: LogItem):  # 添加通信日志
        if (  # 如果是epoch结束且日志不为空且最后一个日志不是epoch结束
            comm_log.is_epoch_end()
            and len(self.comm_logs) > 0
            and not self.comm_logs[-1].is_epoch_end()
        ):
            self.comm_logs.append(comm_log)  # 添加日志
            self.comm_log_each_epoch.append([])  # 添加新的epoch日志列表
            self.epoch_times.append(comm_log.elapsed_time)  # 添加epoch时间
            return
        self.comm_logs.append(comm_log)  # 添加日志
        self.comm_log_each_epoch[-1].append(comm_log)  # 添加到最后一个epoch日志列表

    def analyze(self, print_fn=print):  # 分析日志
        comm_info: Dict[str, Dict] = {}  # 初始化通信信息字典
        _analyze_stage_log(self.comm_log_each_epoch[0], "init", comm_info)  # 分析初始化阶段日志
        for e_log in self.comm_log_each_epoch[1:]:  # 遍历训练阶段日志
            _analyze_stage_log(e_log, "train", comm_info)  # 分析训练阶段日志
        for stage in comm_info.keys():  # 遍历阶段
            if stage != "init":  # 如果不是初始化阶段
                stage_count = comm_info[stage]["count"]  # 获取阶段计数
                comm_type_info = comm_info[stage]["comm_type_info"]  # 获取通信类型信息
                detailed_comm_type_info = comm_info[stage]["detailed_comm_type_info"]  # 获取详细通信类型信息

                log_str = _print_stage_log(stage, stage_count, detailed_comm_type_info, ["comm_type", "comm_group", "msg_size"], ["count"], ["_elapsed_time"], ["busbw"])  # 打印阶段日志
                print_fn(f"\n\tDetailed comm info for AICB {stage} stage\n{log_str}")  # 打印详细通信信息
        return comm_info  # 返回通信信息

    def dump(self, filename):  # 导出日志
        default_comm_folder_path = "results/comm_logs/"  # 默认通信日志文件夹路径
        if not os.path.exists(default_comm_folder_path):  # 如果文件夹不存在
            os.makedirs(default_comm_folder_path, exist_ok=True)  # 创建文件夹
        if "." in filename:  # 如果文件名包含扩展名
            filename = filename.split(".")[0]  # 去掉扩展名
        filename = os.path.join("results/comm_logs/", filename)  # 构建完整文件路径
        csv_filename = filename + "_log.csv"  # 构建CSV文件名
        with open(csv_filename, "w") as f:  # 打开文件
            f.write(self.comm_logs[0].csv_header() + "\n")  # 写入表头
            for log_item in self.comm_logs:  # 遍历日志项
                log_item_write = copy.deepcopy(log_item)  # 深拷贝日志项
                if(log_item_write.comm_type == CommType.computation):  # 如果是计算类型
                    msg_size_str = "("+' '.join(str(shape).replace(',', '') for shape in log_item_write.msg_size)+")"  # 转换消息大小
                    log_item_write.msg_size = msg_size_str  # 更新消息大小
                f.write(log_item_write.view_as_csv_line() + "\n")  # 写入日志行
                del log_item_write  # 删除临时变量
        return csv_filename  # 返回文件名

    @staticmethod  # 定义静态方法
    def load(filename):  # 加载日志
        filename = filename.split(".")  # 分割文件名
        filename[-1] = "pkl"  # 修改扩展名为pkl
        filename = ".".join(filename)  # 重新拼接文件名
        return pickle.load(open(filename, "rb"))  # 加载并返回日志对象

    def _get_elapsed_time(self):  # 获取耗时
        return self.epoch_times  # 返回epoch时间列表

    def analyze_time(self, print_fn=print):  # 分析时间
        self.epoch_times.pop(0)  # 去掉第一个epoch时间
        max_val = max(self.epoch_times)  # 获取最大值
        min_val = min(self.epoch_times)  # 获取最小值
        mean_val = sum(self.epoch_times) / len(self.epoch_times)  # 计算平均值

        variance = sum((x - mean_val) ** 2 for x in self.epoch_times) / len(  # 计算方差
            self.epoch_times
        )
        variance = math.sqrt(variance)  # 计算标准差

        sorted_list = sorted(self.epoch_times)  # 排序时间列表
        p90_val = sorted_list[int(len(sorted_list) * 0.9)]  # 获取90%分位数
        p99_val = sorted_list[int(len(sorted_list) * 0.99)]  # 获取99%分位数
        header = f"{'Init time':<18} {'Max iteration time':<20} {'Min iteration time':<20} {'Avg iteration time':<20} {'P90 iteration time ':<20} {'Iteration time Std ':<20}\n"  # 定义表头
        separator = "-" * len(header) + "\n"  # 定义分隔符
        log_str = separator + header + separator  # 构建日志字符串
        iteration_result = f"{self.epoch_times[0]:<18.2f} {max_val:<20.2f} {min_val:<20.2f} {mean_val:<20.2f} {p90_val:<20.2f} {variance:<20.2f}\n"  # 构建迭代结果
        log_str += iteration_result  # 添加到日志字符串
        print_fn(f"\n\tDetailed info for AICB iteration time\n{log_str}")  # 打印详细信息


class Workload:  # 定义工作负载类
    def __init__(self) -> None:  # 初始化方法
        self.workload = []  # 初始化工作负载列表

    def append(self, log_item: Union[LogItem, Dict]):  # 添加日志项
        if isinstance(log_item, LogItem):  # 如果是LogItem类型
            self.workload.append(log_item)  # 直接添加
            return
        if "stage" not in log_item:  # 如果没有阶段信息
            log_item["stage"] = log_item["operation"] if "operation" in log_item else ""  # 设置阶段信息
        if "comm_group" not in log_item:  # 如果没有通信组信息
            assert (  # 断言通信类型为计算
                log_item["comm_type"] == CommType.computation
            ), "comm_group is required for non-computation comm_type"  # 非计算类型需要通信组
            log_item["comm_group"] = CommGroup.all  # 设置通信组为all
        self.workload.append(  # 添加日志项
            LogItem(
                comm_type=log_item["comm_type"],
                comm_group=log_item["comm_group"],
                comm_group_size=log_item["comm_group_size"],
                msg_size=log_item["msg_size"],
                stage=log_item["stage"],
                src=log_item.get("src", None),
                dst=log_item.get("dst", None),
                additional=log_item.get("additional", None),
            )
        )

    def extend(self, new_workload):  # 扩展工作负载
        self.workload.extend(new_workload.workload)  # 将新工作负载的内容添加到当前工作负载

    def dump(self, filename):  # 导出工作负载
        folder_path = os.path.dirname(filename)  # 获取文件夹路径
        if folder_path and not os.path.exists(folder_path):  # 如果文件夹不存在
            os.makedirs(folder_path)  # 创建文件夹
        default_folder_path = "results/mocked_workload/"  # 默认工作负载文件夹路径
        if not os.path.exists(default_folder_path):  # 如果文件夹不存在
            os.makedirs(default_folder_path, exist_ok=True)  # 创建文件夹
        if "." in filename:  # 如果文件名包含扩展名
            filename = os.path.basename(filename).split(".")[0]  # 去掉扩展名
        filename = os.path.join("results/mocked_workload/", filename)  # 构建完整文件路径
        csv_filename = filename + "_workload.csv"  # 构建CSV文件名
        with open(csv_filename, "w") as f:  # 打开文件
            f.write(self.workload[0].csv_header() + "\n")  # 写入表头
            for log_item in self.workload:  # 遍历日志项
                log_item_write = copy.deepcopy(log_item)  # 深拷贝日志项
                if(log_item_write.comm_type == CommType.computation):  # 如果是计算类型
                    msg_size_str = "("+' '.join(str(shape).replace(',', '') for shape in log_item_write.msg_size)+")"  # 转换消息大小
                    log_item_write.msg_size = msg_size_str  # 更新消息大小
                f.write(log_item_write.view_as_csv_line() + "\n")  # 写入日志行
                del log_item_write  # 删除临时变量
        print(f"Workload file generated:{csv_filename}")  # 打印生成文件信息
        
    @staticmethod  # 定义静态方法
    def load(filename):  # 加载工作负载
        filename = filename.split(".")  # 分割文件名
        filename[-1] = "pkl"  # 修改扩展名为pkl
        filename = ".".join(filename)  # 重新拼接文件名
        workload, args = pickle.load(open(filename, "rb"))  # 加载并返回工作负载和参数
        return workload, args  # 返回工作负载和参数
