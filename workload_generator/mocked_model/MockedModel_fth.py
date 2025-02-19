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

import math  # 引入math模块，提供数学运算功能
from typing import List, Tuple  # 从typing模块导入List和Tuple，分别表示列表和元组类型的注解


class MockedParam:
    def __init__(self, shape: Tuple, elem_size=2, name=None) -> None:  # 构造函数，初始化形状、元素大小和名称
        self.shape = shape  # 参数形状，元组类型，表示维度信息
        self._numel = math.prod(shape)  # 计算元素个数，math.prod计算shape元组的乘积
        self._elem_size = elem_size  # 元素大小，默认为2
        self.name = name if name is not None else "Unknown"  # 参数名称，默认为"Unknown"

    def numel(self):  # 返回参数的元素总数
        return self._numel  # 返回元素个数

    def elem_size(self):  # 返回每个元素的大小
        return self._elem_size  # 返回元素大小

    def msg_size(self):  # 计算消息大小，即参数的总内存大小
        return self._numel * self._elem_size  # 计算总内存大小，元素个数乘以元素大小

    def get_shape(self):  # 返回参数的形状
        return self.shape  # 返回shape属性

    # def name(self):  # 如果需要名字可以去掉注释
    #     return self.param_name  # 返回param_name


def _unpack_params(value: object) -> List[MockedParam]:  # 递归地解包参数，返回MockedParam列表
    if isinstance(value, MockedParam):  # 如果传入的值是MockedParam类型
        return [value]  # 返回包含该MockedParam对象的列表
    elif isinstance(value, MockedModel):  # 如果传入的值是MockedModel类型
        return value.parameters()  # 递归调用parameters()方法，返回该模型的参数列表
    elif isinstance(value, dict):  # 如果传入的值是字典类型
        params = []  # 初始化空列表用于存放参数
        for k, v in value.items():  # 遍历字典的每个键值对
            params += _unpack_params(v)  # 递归地解包字典中的值，并将结果添加到params列表中
        return params  # 返回所有解包的参数列表
    elif isinstance(value, (list, tuple)):  # 如果传入的值是列表或元组类型
        params = []  # 初始化空列表用于存放参数
        for v in value:  # 遍历列表或元组的每个元素
            params += _unpack_params(v)  # 递归解包每个元素
        return params  # 返回所有解包的参数列表
    else:  # 如果传入的值不是上述类型
        return []  # 返回空列表


def _child_modules(value: object) -> List["MockedModel"]:  # 递归地获取子模块，返回MockedModel列表
    if isinstance(value, MockedModel):  # 如果传入的值是MockedModel类型
        modules = [value]  # 将当前MockedModel添加到模块列表中
        modules.extend(_child_modules(value.__dict__))  # 递归获取当前模型的子模块，并将结果添加到modules中
        return modules  # 返回所有子模块
    elif isinstance(value, dict):  # 如果传入的值是字典类型
        modules = []  # 初始化空列表用于存放模块
        for k, v in value.items():  # 遍历字典中的每个键值对
            modules += _child_modules(v)  # 递归地获取字典值中的模块，并将结果添加到modules列表
        return modules  # 返回所有子模块
    elif isinstance(value, (list, tuple)):  # 如果传入的值是列表或元组类型
        modules = []  # 初始化空列表用于存放模块
        for v in value:  # 遍历列表或元组中的每个元素
            modules += _child_modules(v)  # 递归地获取元素中的模块，并将结果添加到modules中
        return modules  # 返回所有子模块
    else:  # 如果传入的值不是上述类型
        return []  # 返回空列表


class MockedModel:
    def __init__(self) -> None:  # 初始化MockedModel类
        self._pre_forward_hook = []  # 前向传播前的钩子函数列表
        self._post_forward_hook = []  # 前向传播后的钩子函数列表
        self._pre_backward_hook = []  # 反向传播前的钩子函数列表
        self._post_backward_hook = []  # 反向传播后的钩子函数列表

    def parameters(self) -> List[MockedParam]:  # 获取当前模型的所有参数
        return _unpack_params(self.__dict__)  # 递归解包模型中的所有参数并返回
 
    def child_modules(self) -> List["MockedModel"]:  # 获取当前模型的所有子模块
        return _child_modules(self.__dict__)  # 递归解包模型中的所有子模块并返回

    def register_forward_pre_hook(self, fn):  # 注册前向传播前的钩子函数
        self._pre_forward_hook.append(fn)  # 将钩子函数添加到_pre_forward_hook列表

    def register_backward_pre_hook(self, fn):  # 注册反向传播前的钩子函数
        self._pre_backward_hook.append(fn)  # 将钩子函数添加到_pre_backward_hook列表

    def register_forward_post_hook(self, fn):  # 注册前向传播后的钩子函数
        self._post_forward_hook.append(fn)  # 将钩子函数添加到_post_forward_hook列表

    def register_backward_post_hook(self, fn):  # 注册反向传播后的钩子函数
        self._post_backward_hook.append(fn)  # 将钩子函数添加到_post_backward_hook列表


class Linear(MockedModel):  # Linear类继承自MockedModel类
    def __init__(self, in_feature, out_feature):  # 初始化Linear类，接受输入特征和输出特征
        self.weight = MockedParam((in_feature, out_feature))  # 创建一个MockedParam对象，表示权重矩阵，其形状为(in_feature, out_feature)
