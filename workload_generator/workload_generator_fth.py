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

from workload_generator.mocked_model.MockedModel import MockedModel  # 导入MockedModel类，用于模拟模型行为
from utils.utils import CommGroup, CommType  # 导入通信组和通信类型相关的工具类
from log_analyzer.log import Workload, LogItem  # 导入日志分析器中的Workload和LogItem类


class WorkloadGenerator:  # 定义一个工作负载生成器类
    # generator = WorkloadGenerator  # 注释掉的代码，可能是为了说明类的实例化方式
    def __init__(self, args, model: MockedModel) -> None:  # 初始化方法，接收参数args和model
        self.name = "workload_generator"  # 设置生成器的名称
        self.args = args  # 将传入的参数args保存为实例变量
        self.model = model  # 将传入的模型model保存为实例变量
        self.workload = Workload()  # 初始化一个Workload对象，用于存储生成的工作负载
        self.epoch = 0  # 初始化epoch计数器

    def __call__(self):  # 定义调用方法，使类实例可以像函数一样被调用
        args = self.args  # 获取初始化时传入的参数args
        self.workload = Workload()  # 重新初始化Workload对象，清空之前的内容
        self.init()  # 调用初始化方法，可能用于设置一些初始状态
        self.workload.append(LogItem(comm_type=CommType.epoch_end))  # 添加一个epoch结束的日志项
        for i in range(args.epoch_num):  # 遍历epoch数量，生成每个epoch的工作负载
            if args.pipeline_model_parallel > 1 and args.frame != "collective_test":  # 如果启用了流水线并行且框架不是collective_test
                self.with_pipeline_forward_backward()  # 调用流水线前向和反向传播方法
                self.step()  # 调用step方法，可能用于更新模型参数
            else:  # 如果未启用流水线并行或框架是collective_test
                for _ in range(args.num_microbatches):  # 遍历微批次数量
                    self.forward()  # 调用前向传播方法
                    self.backward()  # 调用反向传播方法
            self.step()  # 调用step方法，可能用于更新模型参数
            self.workload.append(LogItem(comm_type=CommType.epoch_end))  # 添加一个epoch结束的日志项
        return self.workload  # 返回生成的工作负载

    def forward(self):  # 定义前向传播方法
        pass  # 空实现，具体逻辑需由子类或后续代码补充

    def backward(self):  # 定义反向传播方法
        pass  # 空实现，具体逻辑需由子类或后续代码补充

    def step(self):  # 定义step方法，可能用于更新模型参数
        pass  # 空实现，具体逻辑需由子类或后续代码补充

    def with_pipeline_forward_backward(self):  # 定义流水线前向和反向传播方法
        pass  # 空实现，具体逻辑需由子类或后续代码补充