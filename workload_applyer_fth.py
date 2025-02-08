"""
Copyright (c) 2021, Alibaba Group; # 版权声明，归属阿里巴巴集团
Licensed under the Apache License, Version 2.0 (the "License"); # 使用 Apache 2.0 许可证
you may not use this file except in compliance with the License. # 除非符合许可证要求，否则不得使用此文件
You may obtain a copy of the License at # 获取许可证副本的地址
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software # 除非法律要求或书面同意，软件
distributed under the License is distributed on an "AS IS" BASIS, # 根据“现状”分发，不提供任何形式的担保
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. # 不保证任何明示或暗示的条件
See the License for the specific language governing permissions and # 查看许可证以了解具体的权限和限制
limitations under the License.
"""
import torch # 导入 PyTorch 库
import sys # 导入系统模块
import math # 导入数学模块
import time # 导入时间模块
from utils.utils import WorkloadWriter, CommGroup, CommType, ReduceOp # 从工具模块导入自定义类和函数
from utils.benchmark_logger import bench_logger # 从日志模块导入基准测试日志记录器
import utils.utils as utils # 导入工具模块


class WorkloadApplyer: # 定义工作负载应用器类
    def __init__(self, workload=None, args=None, filename=None) -> None: # 初始化方法
        if workload is None or args is None: # 如果 workload 或 args 为空
            assert (
                filename is None
            ), f"you should either pass workload,args or filename to init WorkloadApplyer" # 确保必须传递 workload、args 或 filename
            workload, args = WorkloadWriter.load_workload(filename) # 从文件加载工作负载和参数
        # if not hasattr(args, "backend"): # 如果 args 没有 backend 属性
        #     args.backend = "nccl" # 设置默认后端为 nccl
        # torch.distributed.init_process_group(backend=args.backend) # 初始化分布式进程组
        self.args = args # 将参数赋值给实例变量
        world_size = torch.distributed.get_world_size() # 获取全局进程数
        # args.rank = torch.distributed.get_rank() # 获取当前进程的 rank
        if args.world_size != world_size: # 如果生成工作负载时的世界大小与当前世界大小不同
            print(
                f"WARNNING: world_size is {args.world_size} when generating workload, but now world size is {world_size}"
            ) # 打印警告信息
            args.world_size = torch.distributed.get_world_size() # 更新世界大小
        device_count = torch.cuda.device_count() # 获取 GPU 设备数量
        self.device = args.rank % device_count # 计算当前设备编号
        torch.cuda.set_device(self.device) # 设置当前设备
        self.device = torch.cuda.current_device() # 获取当前设备编号
        self.comm_group_info, self.pp_global_rank_info = (
            self._generate_dp_tp_pp_ep_groups()
        ) # 生成通信组信息和流水线并行全局 rank 信息
        self.workload = workload # 将工作负载赋值给实例变量
        self.comm_type_function = { # 定义通信类型对应的操作函数
            CommType.barrier: self._apply_barrier, # 阻塞操作
            CommType.broadcast: self._apply_broadcast, # 广播操作
            CommType.reduce: self._apply_reduce, # 规约操作
            CommType.all_reduce: self._apply_all_reduce, # 全规约操作
            CommType.all_gather: self._apply_all_gather, # 全收集操作
            CommType.reduce_scatter: self._apply_reduce_scatter, # 规约分散操作
            CommType.isend: self._apply_p2pcommunication, # 点对点发送操作
            CommType.irecv: self._apply_p2pcommunication, # 点对点接收操作
            CommType.all_gather_into_tensor: self._apply_all_gather, # 全收集到张量操作
            CommType.reduce_scatter_tensor: self._apply_reduce_scatter, # 规约分散到张量操作
            CommType.computation: self._apply_computation, # 计算操作
            CommType.all_to_all: self._apply_all_to_all, # 全交换操作
            CommType.epoch_end: bench_logger.end_epoch, # 结束 epoch 操作

        }

        cal_tuple_num = lambda t: math.prod(t[0]) + math.prod(t[1]) # 定义计算元组大小的函数
        max_msg_size = max(
            [
                (
                    item.msg_size
                    if isinstance(item.msg_size, int)
                    else cal_tuple_num(item.msg_size)
                )
                for item in self.workload.workload
            ]
        ) # 计算最大消息大小
        self.gemm_cache = {} # 初始化 GEMM 缓存
        self.computation_aiob = False # 初始化 AIOB 计算标志
        if args.aiob_enable and args.frame == "Megatron": # 如果启用 AIOB 且框架为 Megatron
            self.computation_aiob = True # 设置 AIOB 计算标志为 True

        self.skip_computation = False # 是否跳过计算
        self.always_apply_gemm = False # 是否始终应用 GEMM
        self.gemm_iters = 1 if self.always_apply_gemm else 50 # 设置 GEMM 迭代次数
        self.buffer = torch.empty(
            (max_msg_size,), dtype=torch.bfloat16, device=self.device
        ) # 创建缓冲区张量

    def _generate_dp_tp_pp_ep_groups(self): # 生成数据并行、张量并行、流水线并行、专家并行组
        """Borrow from Megatron-LM""" # 借助 Megatron-LM 的实现
        all_data_parallel_group_ranks = [] # 初始化数据并行组 rank 列表
        world_size = self.args.world_size # 获取世界大小
        rank = torch.distributed.get_rank() # 获取当前 rank
        self.rank = rank # 将 rank 赋值给实例变量
        tensor_model_parallel_size, pipeline_model_parallel_size, data_parallel_size,expert_model_parallel_size = (
            self.args.tensor_model_parallel_size,
            self.args.pipeline_model_parallel,
            self.args.dp_num,
            self.args.expert_model_parallel_size,
        ) # 获取张量并行、流水线并行、数据并行、专家并行大小
        rank_generator = utils.RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=self.args.context_parallel_size,
        order='tp-cp-ep-dp-pp',
    ) # 创建 Rank 生成器
        for ranks in rank_generator.get_ranks('ep', independent_ep=True): # 遍历专家并行组 rank
            group = torch.distributed.new_group(
                ranks
            ) # 创建新的通信组
            if rank in ranks:
                ep_group = group # 如果当前 rank 在组中，保存专家并行组
        for ranks in rank_generator.get_ranks('tp'): # 遍历张量并行组 rank
            group = torch.distributed.new_group(
                ranks
            ) # 创建新的通信组
            if rank in ranks:
                tp_group = group # 如果当前 rank 在组中，保存张量并行组
        for ranks in rank_generator.get_ranks('pp'): # 遍历流水线并行组 rank
            group = torch.distributed.new_group(
                ranks
            ) # 创建新的通信组
            if rank in ranks:
                pp_group = group # 如果当前 rank 在组中，保存流水线并行组
                pp_global_rank = ranks # 保存流水线并行全局 rank
        for ranks in rank_generator.get_ranks('dp'): # 遍历数据并行组 rank
            group = torch.distributed.new_group(
                ranks
            ) # 创建新的通信组
            if rank in ranks:
                dp_group = group # 如果当前 rank 在组中，保存数据并行组
        for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True): # 遍历张量-专家并行组 rank
            group = torch.distributed.new_group(
                ranks
            ) # 创建新的通信组
            if rank in ranks:
                ep_tp_group = group # 如果当前 rank 在组中，保存张量-专家并行组
        for ranks in rank_generator.get_ranks('dp', independent_ep=True): # 遍历数据-专家并行组 rank
            group = torch.distributed.new_group(
                ranks
            ) # 创建新的通信组
            if rank in ranks:
                ep_dp_group = group # 如果当前 rank 在组中，保存数据-专家并行组
        return {
            CommGroup.tp_group: tp_group, # 返回张量并行组
            CommGroup.dp_group: dp_group, # 返回数据并行组
            CommGroup.pp_group: pp_group, # 返回流水线并行组
            CommGroup.ep_group: ep_group, # 返回专家并行组
            CommGroup.ep_tp_group: ep_tp_group, # 返回张量-专家并行组
            CommGroup.ep_dp_group: ep_dp_group, # 返回数据-专家并行组
        }, pp_global_rank # 返回流水线并行全局 rank

    def _get_pipeline_parallel_size(self): # 获取流水线并行大小
        group = self.comm_group_info["pp_group"] # 获取流水线并行组
        pp_group_size = torch.distributed.get_world_size(group) # 获取组大小
        return pp_group_size # 返回流水线并行大小

    def _get_pipeline_parallel_rank(self): # 获取流水线并行 rank
        group = self.comm_group_info["pp_group"] # 获取流水线并行组
        pp_rank = torch.distributed.get_rank(group) # 获取组内 rank
        return pp_rank # 返回流水线并行 rank

    def _get_pipeline_prev_rank(self): # 获取前一个流水线并行 rank
        rank_in_pipeline = self._get_pipeline_parallel_rank() # 获取当前流水线并行 rank
        world_size = self._get_pipeline_parallel_size() # 获取流水线并行大小
        return self.pp_global_rank_info[(rank_in_pipeline - 1) % world_size] # 返回前一个 rank

    def _get_pipeline_next_rank(self): # 获取下一个流水线并行 rank
        rank_in_pipeline = self._get_pipeline_parallel_rank() # 获取当前流水线并行 rank
        world_size = self._get_pipeline_parallel_size() # 获取流水线并行大小
        return self.pp_global_rank_info[(rank_in_pipeline + 1) % world_size] # 返回下一个 rank

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_p2pcommunication(self, item): # 应用点对点通信
        ops = [] # 初始化操作列表
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2) # 获取缓冲区切片
        if item.additional == "send_prev": # 如果需要向前发送
            if self._get_pipeline_parallel_rank() != 0: # 如果不是第一个流水线阶段
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor, self._get_pipeline_prev_rank()
                ) # 创建发送操作
                ops.append(send_prev_op) # 添加到操作列表
            else:
                pass
        if item.additional == "send_next": # 如果需要向后发送
            if self._get_pipeline_parallel_rank() != self.args.pipeline_model_parallel - 1: # 如果不是最后一个流水线阶段
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor, self._get_pipeline_next_rank()
                ) # 创建发送操作
                ops.append(send_next_op) # 添加到操作列表
            else:
                pass
        if item.additional == "recv_prev": # 如果需要从前一个接收
            if self._get_pipeline_parallel_rank() != 0: # 如果不是第一个流水线阶段
                tensor_recv_prev = torch.empty(
                    item.msg_size // 2, dtype=torch.bfloat16, device=self.device
                ) # 创建接收张量
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    tensor_recv_prev,
                    self._get_pipeline_prev_rank(),
                ) # 创建接收操作
                ops.append(recv_prev_op) # 添加到操作列表
            else:
                pass
        if item.additional == "recv_next": # 如果需要从下一个接收
            if self._get_pipeline_parallel_rank() != self.args.pipeline_model_parallel - 1: # 如果不是最后一个流水线阶段
                tensor_recv_next = torch.empty(
                    item.msg_size // 2, dtype=torch.bfloat16, device=self.device
                ) # 创建接收张量
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv,
                    tensor_recv_next,
                    self._get_pipeline_next_rank(),
                ) # 创建接收操作
                ops.append(recv_next_op) # 添加到操作列表
            else:
                pass
        if len(ops) > 0: # 如果有操作
            reqs = torch.distributed.batch_isend_irecv(ops) # 批量执行发送和接收操作
            for req in reqs:
                req.wait() # 等待所有操作完成

        torch.cuda.synchronize() # 同步 CUDA 流

    def _apply_barrier(self, item): # 应用阻塞操作
        torch.distributed.barrier() # 执行分布式阻塞

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_broadcast(self, item): # 应用广播操作
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2) # 获取缓冲区切片
        group = self.comm_group_info[item.comm_group] # 获取通信组
        src = torch.distributed.get_global_rank(group, 0) # 获取源 rank
        return torch.distributed.broadcast(
            tensor=tensor, src=src, group=group, async_op=False
        ) # 执行广播操作

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_reduce(self, item): # 应用规约操作
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2) # 获取缓冲区切片
        group = self.comm_group_info[item.comm_group] # 获取通信组
        dst = item.dst # 获取目标 rank
        return torch.distributed.reduce(
            tensor=tensor,
            dst=dst,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
            async_op=False,
        ) # 执行规约操作

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_all_reduce(self, item): # 应用全规约操作
        tensor = torch.narrow(self.buffer, 0, 0, item.msg_size // 2) # 获取缓冲区切片
        group = self.comm_group_info[item.comm_group] # 获取通信组
        return torch.distributed.all_reduce(
            tensor=tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
            async_op=False,
        ) # 执行全规约操作

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_all_gather(self, item): # 应用全收集操作
        group = self.comm_group_info[item.comm_group] # 获取通信组
        num_elements = item.msg_size // 2 # 计算元素数量
        padding_size = (
            (group.size() - num_elements % group.size())
            if num_elements % group.size()
            else 0
        ) # 计算填充大小
        num_elements = num_elements + padding_size # 更新元素数量
        output_tensor = torch.narrow(self.buffer, 0, 0, num_elements) # 获取输出张量
        input_tensor_size = output_tensor.numel() // group.size() # 计算输入张量大小
        group_rank = torch.distributed.get_group_rank(group, self.rank) # 获取组内 rank
        input_tensor = torch.narrow(
            output_tensor, 0, group_rank * input_tensor_size, input_tensor_size
        ) # 获取输入张量
        return torch.distributed.all_gather_into_tensor(
            output_tensor, input_tensor, group=group, async_op=False
        ) # 执行全收集操作

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _overlap(self, item): # 应用重叠操作
        item.additional = 'overlap' # 设置附加属性为 overlap

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_reduce_scatter(self, item): # 应用规约分散操作
        group = self.comm_group_info[item.comm_group] # 获取通信组
        num_elements = item.msg_size // 2 # 计算元素数量
        padding_size = (
            (group.size() - num_elements % group.size())
            if num_elements % group.size()
            else 0
        ) # 计算填充大小
        num_elements = num_elements + padding_size # 更新元素数量
        input_tensor = torch.narrow(self.buffer, 0, 0, num_elements) # 获取输入张量
        group = self.comm_group_info[item.comm_group] # 获取通信组
        output_tensor_size = input_tensor.numel() // group.size() # 计算输出张量大小
        group_rank = torch.distributed.get_group_rank(group, self.rank) # 获取组内 rank
        output_tensor = torch.narrow(
            input_tensor, 0, group_rank * output_tensor_size, output_tensor_size
        ) # 获取输出张量
        return torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, group=group, async_op=False
        ) # 执行规约分散操作

    @bench_logger.log_timing("comm") # 日志记录通信时间
    def _apply_all_to_all(self, item): # 应用全交换操作
        group = self.comm_group_info[item.comm_group] # 获取通信组
        num_elements = item.msg_size // 2 # 计算元素数量
        input_tensor = torch.narrow(self.buffer, 0, 0, num_elements) # 获取输入张量
        output_tensor = torch.empty(
            num_elements * group.size(),
            dtype=self.buffer.dtype,
            device=self.buffer.device,
        ) # 创建输出张量
        return torch.distributed.all_to_all_single(
            output_tensor, input_tensor, group=group
        ) # 执行全交换操作

    @bench_logger.log_timing("comp") # 日志记录计算时间
    def _apply_computation(self, item): # 应用计算操作
        if self.skip_computation: # 如果跳过计算
            return
        if self.computation_aiob: # 如果启用了 AIOB
            time.sleep(item._elapsed_time/ 1e9) # 模拟计算耗时
        else:
            input_shape1, input_shape2 = item.msg_size # 获取输入形状
            A, B = torch.rand(input_shape1, device=self.device), torch.rand(
                input_shape2, device=self.device
            ) # 创建随机张量
            torch.matmul(A, B) # 执行矩阵乘法
            return

    def apply_workload(self): # 应用工作负载
        torch.cuda.synchronize(self.device) # 同步 CUDA 流
        start = time.perf_counter() # 记录开始时间
        key = "backward" # 定义关键字
        for item in self.workload.workload: # 遍历工作负载
            if (
                self.computation_aiob
                and item.comm_type == CommType.all_reduce
                and key in item.stage
            ): # 如果启用了 AIOB 且通信类型为全规约且阶段包含关键字
                comm_func = self.comm_type_function[item.comm_type] # 获取通信函数
            else:
                comm_func = self.comm_type_function[item.comm_type] # 获取通信函数
                comm_func(item) # 执行通信操作
        torch.cuda.synchronize(self.device) # 同步 CUDA 流
        end = time.perf_counter() # 记录结束时间
        return end - start # 返回总耗时


if __name__ == "__main__": # 主程序入口
    filename = "results/model_workload/local_deepspeed_stage3.csv" # 定义工作负载文件路径
    applyer = WorkloadApplyer(filename=filename) # 创建工作负载应用器实例
    applyer.apply_workload() # 应用工作负载
    if torch.distributed.get_rank() == 0: # 如果是主进程
        bench_logger.analyze_comm_log(bench_logger.comm_log) # 分析通信日志
