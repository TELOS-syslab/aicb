```python
# 版权声明，遵循Apache License 2.0开源协议
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

# 导入所需的模块和类
from utils.utils import CommType, CommGroup, get_params, WorkloadWriter  # 导入通信类型、通信组、获取参数和工作负载写入器
from log_analyzer.log import LogItem, Workload  # 导入日志项和工作负载类
from workload_generator.workload_generator import WorkloadGenerator  # 导入工作负载生成器基类


# 定义Collective_Test类，继承自WorkloadGenerator
class Collective_Test(WorkloadGenerator):
    # 初始化方法，接收参数args和模型model
    def __init__(self, args, model):
        super().__init__(args, model)  # 调用父类的初始化方法
        self.args = args  # 存储传入的参数
        self.name = "collective_test"  # 实例变量，表示生成器的名称

    # 初始化方法，用于生成预热阶段的工作负载
    def init(self):
        iter_num = self.args.iter_num  # 获取迭代次数
        for i in range(iter_num):  # 遍历迭代次数
            # 添加预热阶段的日志项
            self.workload.append(
                LogItem(
                    comm_type=CommType.get_comm_type(self.args.test_comm),  # 获取通信类型
                    comm_group=CommGroup.dp_group,  # 设置通信组为数据并行组
                    comm_group_size=self.args.dp_num,  # 设置通信组大小为数据并行数
                    msg_size=self.args.begin_size,  # 设置消息大小为初始大小
                    stage="warmup",  # 设置阶段为预热
                )
            )

    # 步进方法，用于生成测试阶段的工作负载
    def step(self):
        test_comm = CommType.get_comm_type(self.args.test_comm)  # 获取测试通信类型
        begin_size = self.args.begin_size  # 获取初始消息大小
        end_size = self.args.end_size  # 获取结束消息大小
        curr_size = begin_size  # 当前消息大小初始化为初始大小
        iter_num = self.args.iter_num  # 获取迭代次数
        multi_all_reduce_enable = self.args.multi_all_reduce_enable  # 获取是否启用多AllReduce的配置

        while curr_size <= end_size:  # 当前消息大小小于等于结束大小时循环
            # self.workload.append(LogItem(comm_type=CommType.epoch_end))
            if not multi_all_reduce_enable:  # 如果未启用多AllReduce
                for i in range(iter_num):  # 遍历迭代次数
                    self.workload.append(
                        LogItem(
                            comm_type=test_comm,  # 设置通信类型为测试通信类型
                            comm_group=CommGroup.dp_group,  # 设置通信组为数据并行组
                            comm_group_size=self.args.dp_num,  # 设置通信组大小为数据并行数
                            msg_size=curr_size,  # 设置消息大小为当前大小
                            stage="test_step",  # 设置阶段为测试步骤
                        )
                    )
                curr_size *= 2  # 当前消息大小乘以2
            else:  # 如果启用了多AllReduce
                for i in range(iter_num):  # 遍历迭代次数
                    self.workload.append(
                        LogItem(
                            comm_type=test_comm,  # 设置通信类型为测试通信类型
                            comm_group=CommGroup.pp_group,  # 设置通信组为管道并行组
                            comm_group_size=self.args.pipeline_model_parallel,  # 设置通信组大小为管道并行数
                            msg_size=curr_size,  # 设置消息大小为当前大小
                            stage="test_step",  # 设置阶段为测试步骤
                        )
                    )
                curr_size *= 2  # 当前消息大小乘以2


# 主程序入口
if __name__ == "__main