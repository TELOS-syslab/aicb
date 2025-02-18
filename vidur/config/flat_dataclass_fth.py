import json
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    BooleanOptionalAction,
)
from collections import defaultdict, deque
from dataclasses import MISSING, fields, make_dataclass
from typing import Any, get_args

from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.utils import (
    get_all_subclasses,
    get_inner_type,
    is_bool,
    is_composed_of_primitives,
    is_dict,
    is_list,
    is_optional,
    is_primitive_type,
    is_subclass,
    to_snake_case,
)


def topological_sort(dataclass_dependencies: dict) -> list:
    # 对数据类的依赖关系进行拓扑排序，返回有序类的列表
    in_degree = defaultdict(int)
    # 默认字典，用于存储每个类的入度（依赖数量）
    for cls, dependencies in dataclass_dependencies.items():
        # 遍历每个类及其依赖项
        for dep in dependencies:
            in_degree[dep] += 1
            # 增加依赖类的入度

    zero_in_degree_classes = deque(
        [cls for cls in dataclass_dependencies if in_degree[cls] == 0]
        # 创建入度为0的类的双端队列
    )
    sorted_classes = []
    # 用于存储排序后的类

    while zero_in_degree_classes:
        # 若存在入度为0的类
        cls = zero_in_degree_classes.popleft()
        # 弹出一个类
        sorted_classes.append(cls)
        # 将其添加到排序结果的列表中
        for dep in dataclass_dependencies[cls]:
            # 减少依赖类的入度
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                # 如果减至0，则将其添加到入度为0的类队列中
                zero_in_degree_classes.append(dep)

    return sorted_classes
    # 返回排序后的类列表


def reconstruct_original_dataclass(self) -> Any:
    # 重构原始数据类
    """
    This function is dynamically mapped to FlatClass as an instance method.
    """
    sorted_classes = topological_sort(self.dataclass_dependencies)
    # 对数据类的依赖关系进行拓扑排序
    instances = {}
    # 存储类的实例

    for _cls in reversed(sorted_classes):
        # 逆序遍历排序后的类
        args = {}

        for prefixed_field_name, original_field_name, field_type in self.dataclass_args[
            _cls
        ]:
            # 遍历数据类的参数信息
            if is_subclass(field_type, BasePolyConfig):
                # 如果字段类型是BasePolyConfig的的子类
                config_type = getattr(self, f"{original_field_name}_type")
                # 获取字段的配置类型
                # 查找字段类型的所有子类，并检查哪一个与配置类型匹配
                config_type_matched = False
                for subclass in get_all_subclasses(field_type):
                    if str(subclass.get_type()) == config_type:
                        # 子类配置类型匹配
                        config_type_matched = True
                        args[original_field_name] = instances[subclass]
                        break
                assert (
                    config_type_matched
                ), f"Invalid type {config_type} for {prefixed_field_name}_type. Valid types: {[str(subclass.get_type()) for subclass in get_all_subclasses(field_type)]}"
                # 断言确保匹配成功，否则报错
            elif hasattr(field_type, "__dataclass_fields__"):
                # 如果字段类型是数据类
                args[original_field_name] = instances[field_type]
            else:
                value = getattr(self, prefixed_field_name)
                # 获取字段值
                if callable(value):
                    # 处理可调用的默认工厂值
                    value = value()
                args[original_field_name] = value

        instances[_cls] = _cls(**args)
        # 创建类的实例

    return instances[sorted_classes[0]]
    # 返回构建的第一个类实例


@classmethod
def create_from_cli_args(cls) -> Any: # 从命令行参数创建实例
    """
    This function is dynamically mapped to FlatClass as a class method. 这函数被动态映射为 FlatClass 的一个类方法。
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter) # 创建命令行参数解析器

    for field in fields(cls): # 遍历类的字段
        nargs = None
        action = None
        field_type = field.type
        help_text = cls.metadata_mapping[field.name].get("help", None) # 得到字段类型和帮助文本

        if is_list(field.type): # 如果字段是列表类型
            assert is_composed_of_primitives(field.type) # 确认列表由原始类型组成
            field_type = get_args(field.type)[0] # 获取列表中元素的类型
            if is_primitive_type(field_type):
                nargs = "+" # 设置可选参数
            else:
                field_type = json.loads
        elif is_dict(field.type): # 如果字段是字典类型
            assert is_composed_of_primitives(field.type)
            field_type = json.loads # 将字段类型设置为JSON解析器
        elif is_bool(field.type): # 如果字段是布尔类型
            action = BooleanOptionalAction # 设置参数操作为布尔可选的

        arg_params = {
            "type": field_type,
            "action": action,
            "help": help_text, # 配置参数的类型、操作和帮助文本
        }

        # 处理具有默认值和默认工厂的参数
        if field.default is not MISSING:
            # 如果字段有默认值
            value = field.default
            if callable(value):
                value = value() # 确保返回实际值
            arg_params["default"] = value
        elif field.default_factory is not MISSING: # 如果有默认工厂
            arg_params["default"] = field.default_factory()
        else:
            arg_params["required"] = True # 默认情况下必需

        if nargs:
            arg_params["nargs"] = nargs # 如果有选择参数则添加
        parser.add_argument(f"--{field.name}", **arg_params) # 添加参数至命令行解析器

    args = parser.parse_args() # 解析命令行参数

    return cls(**vars(args)) # 通过解析得到的参数字典创建类实例


def create_flat_dataclass(input_dataclass: Any) -> Any:
    # 创建一个平面数据类
    """
    Creates a new FlatClass type by recursively flattening the input dataclass.
    This allows for easy parsing of command line arguments along with storing/loading the configuration to/from a file.
    通过递归展平输入的数据类来创建一个新的 FlatClass 类型。这便于解析命令行参数，并能够将配置存储到文件或从文件加载。
    """
    meta_fields_with_defaults = []
    meta_fields_without_defaults = [] # 元数据字段，分为有默认值和没有的
    processed_classes = set() # 存储处理过的类
    dataclass_args = defaultdict(list)
    dataclass_dependencies = defaultdict(set) # 数据类的参数和依赖关系
    metadata_mapping = {} # 字段的元数据信息

    def process_dataclass(_input_dataclass, prefix=""):
        if _input_dataclass in processed_classes: # 如果类已经处理过，则返回
            return

        processed_classes.add(_input_dataclass) # 将类添加到已处理集合

        for field in fields(_input_dataclass): # 遍历类的字段
            prefixed_name = f"{prefix}{field.name}" # 生成带前缀的字段名

            if is_optional(field.type): # 如果字段类型是可选的
                field_type = get_inner_type(field.type) # 获取内部类型
            else:
                field_type = field.type

            # 如果字段是BasePolyConfig的子类，处理它作为数据类
            if is_subclass(field_type, BasePolyConfig):
                dataclass_args[_input_dataclass].append(
                    (field.name, field.name, field_type) # 将字段信息增加到参数列表
                )

                type_field_name = f"{field.name}_type" # 添加一个类型参数
                default_value = str(field.default_factory().get_type()) # 获取默认值
                meta_fields_with_defaults.append(
                    (type_field_name, type(default_value), default_value)
                )
                metadata_mapping[type_field_name] = field.metadata

                assert hasattr(field_type, "__dataclass_fields__") # 断言确认是数据类
                for subclass in get_all_subclasses(field_type): # 遍历字段类型的所有子类
                    dataclass_dependencies[_input_dataclass].add(subclass) # 增加依赖关系
                    process_dataclass(subclass, f"{to_snake_case(subclass.__name__)}_")
                continue

            # 如果字段是数据类，递归处理
            if hasattr(field_type, "__dataclass_fields__"):
                dataclass_dependencies[_input_dataclass].add(field_type)
                dataclass_args[_input_dataclass].append(
                    (field.name, field.name, field_type)
                )
                process_dataclass(field_type, f"{to_snake_case(field_type.__name__)}_")
                continue

            field_default = field.default if field.default is not MISSING else MISSING
            field_default_factory = (
                field.default_factory
                if field.default_factory is not MISSING
                else MISSING
            )
            # 收集字段的默认值和默认工厂

            if field_default is not MISSING: # 如果字段有默认值
                meta_fields_with_defaults.append(
                    (prefixed_name, field_type, field_default) # 将其加入有默认值的元数据字段列表中
                )
            elif field_default_factory is not MISSING: # 如果字段有默认工厂
                meta_fields_with_defaults.append(
                    (prefixed_name, field_type, field_default_factory)
                )
            else: # 没有默认值或默认工厂
                meta_fields_without_defaults.append((prefixed_name, field_type))

            dataclass_args[_input_dataclass].append(
                (prefixed_name, field.name, field_type) # 增加参数信息
            )
            metadata_mapping[prefixed_name] = field.metadata

    process_dataclass(input_dataclass) # 处理输入的数据类

    meta_fields = meta_fields_without_defaults + meta_fields_with_defaults # 合并字段（有默认值和没有默认值）
    FlatClass = make_dataclass("FlatClass", meta_fields) # 创建一个新的平面数据类

    # Metadata fields
    FlatClass.dataclass_args = dataclass_args
    FlatClass.dataclass_dependencies = dataclass_dependencies
    FlatClass.metadata_mapping = metadata_mapping

    # Helper methods
    FlatClass.reconstruct_original_dataclass = reconstruct_original_dataclass
    FlatClass.create_from_cli_args = create_from_cli_args # 为平面类增加帮助方法

    return FlatClass # 返回生成的平面数据类
