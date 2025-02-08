"""Copyright和License信息，此处无需修改"""
#!/bin/python
"""运行megatron on GPT-7B的示例"""
from utils.utils import CommGroup, CommType, get_params, WorkloadWriter  # 导入通信相关工具类
from workload_generator.workload_generator import WorkloadGenerator  # 工作负载生成器基类
from workload_generator.mocked_model.MockedMegatron import MegatronModel  # Mock的Megatron模型
from log_analyzer.log import LogItem  # 日志项定义


class MegatronWorkload(WorkloadGenerator):
    def __init__(self, args, model):
        super().__init__(args, model)  # 调用父类初始化
        self.name = "megatron"  # 工作负载名称
        self.args = args  # 命令行参数
        self.tp_is_enable = True if args.tensor_model_parallel_size > 1 else False  # 判断是否启用张量并行

    def _get_total_params(self):
        """计算模型总参数量"""
        total_params = 0
        for param in self.model.parameters():
            total_params += param.numel()  # 累加所有参数的数目
        return total_params

    def _get_layernorm_params(self):
        """获取层归一化层的参数量（用于序列并行）"""
        total_params = 0
        for param in self.model.parameters():
            if getattr(param, "sequence_parallel", False):  # 检查是否启用序列并行
                total_params += param.numel()
        return total_params

    def init(self):
        """初始化阶段的通信操作记录"""
        args = self.args
        # 模型初始化阶段的四次all_reduce操作
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,  # 数据并行组
                comm_group_size=self.args.dp_num,  # 数据并行组大小
                msg_size=1 * 8,  # 消息大小(单位字节)
                stage="init.model_setup",
            )
        )
        for _ in range(3):  # 后续三次all_reduce
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=1 * 8,
                    stage="init.model_setup",
                )
            )
            if args.pipeline_model_parallel > 1:  # 流水线并行时添加pp组的all_reduce
                self.workload.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.pp_group,  # 流水线并行组
                        comm_group_size=self.args.pipeline_model_parallel,
                        msg_size=1 * 8,
                        stage="init.model_setup",
                    )
                )
        # 时间同步的all_gather操作
        self.workload.append(
            LogItem(
                comm_type=CommType.all_gather,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=4 * 8,  # 4个8字节数据
                stage="init.model_setup",
            )
        )
        # 张量并行组的广播操作
        self.workload.append(
            LogItem(
                comm_type=CommType.broadcast,
                comm_group=CommGroup.tp_group,  # 张量并行组
                comm_group_size=self.args.tensor_model_parallel_size,
                msg_size=3 * 8,  # 3个8字节数据
                stage="init.model_setup",
                src=0,  # 广播源
            )
        )
        # 最后一个流水线阶段的embedding参数all_reduce
        if args.pp_rank == args.pipeline_model_parallel - 1 and args.pipeline_model_parallel > 1:
            for p in self.model.embedding.parameters():
                self.workload.append(
                    LogItem(
                        comm_type=CommType.all_reduce,
                        comm_group=CommGroup.tp_group,
                        comm_group_size=self.args.tensor_model_parallel_size,
                        msg_size=p.msg_size(),  # 参数大小
                        stage="init.model_setup",
                    )
                )
        # 最终的时间同步all_gather
        self.workload.append(
            LogItem(
                comm_type=CommType.all_gather,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=8 * 8,  # 8个8字节数据
                stage="init.model_setup",
            )
        )

    def get_pp_rank(self, rank, world_size, pp_size):
        """计算在流水线并行中的rank位置"""
        ranks_per_pp_group = world_size // pp_size  # 每个流水线组的rank数
        pp_rank = rank // ranks_per_pp_group  # 当前rank的流水线组内位置
        return pp_rank

    def with_pipeline_forward_backward(self):
        """处理流水线并行的前向和反向传播通信"""
        args = self.args
        # 获取当前rank的流水线阶段
        if args.workload_only:
            rank = 0  # 仅生成工作负载时使用0号rank
        else:
            import torch
            rank = torch.distributed.get_rank()  # 实际运行时获取rank
        world_size = args.world_size
        pp_rank = self.get_pp_rank(rank, world_size, args.pipeline_model_parallel)
        # 计算预热和剩余的microbatch数量
        pp_num_warmup_microbatches = min(
            args.pipeline_model_parallel - pp_rank - 1, args.num_microbatches
        )
        num_microbatches_remaining = args.num_microbatches - pp_num_warmup_microbatches

        # 前向传播预热阶段
        for _ in range(pp_num_warmup_microbatches):
            if pp_rank != 0:  # 非第一个流水线阶段需要接收前驱数据
                self.workload.append(
                    LogItem(
                        comm_type=CommType.irecv,  # 异步接收
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),  # 输入数据大小
                        stage="forward_step",
                        additional="recv_prev",
                    )
                )
            # 张量并行组的参数广播
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=5 * 8,  # 5个8字节参数
                    stage="forward_step",
                    src=0,
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=8 * (args.world_size + args.seq_length * args.micro_batch),  # 广播数据大小
                    stage="forward_step",
                    src=0,
                )
            )
            # 前向传播计算
            self.workload.extend(self.model.forward())  # 添加模型前向传播的通信操作
            # 非最后一个流水线阶段需要发送数据给后继
            if pp_rank != args.pipeline_model_parallel - 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.isend,  # 异步发送
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="send_next",
                    )
                )

        # 处理剩余的microbatch
        if num_microbatches_remaining > 0 and pp_rank != 0:
            self.workload.append(
                LogItem(
                    comm_type=CommType.irecv,
                    comm_group=CommGroup.pp_group,
                    comm_group_size=1,
                    msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                    stage="forward_step",
                    additional="recv_prev",
                )
            )

        # 处理剩余microbatch的前向和反向传播
        for i in range(num_microbatches_remaining):
            last_iter = i == (num_microbatches_remaining - 1)  # 是否最后一个迭代
            # 参数广播
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=5 * 8,
                    stage="forward_step",
                    src=0,
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=8 * (args.world_size + args.seq_length * args.micro_batch),
                    stage="forward_step",
                    src=0,
                )
            )
            # 前向传播
            self.workload.extend(self.model.forward())
            # 非最后阶段需要处理流水线通信
            if pp_rank != args.pipeline_model_parallel - 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.irecv,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="recv_next",
                    )
                )
                self.workload.append(
                    LogItem(
                        comm_type=CommType.isend,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="forward_step",
                        additional="send_next",
                    )
                )
            # 反向传播
            self.workload.extend(self.model.backward())

            # 处理反向传播的流水线通信
            if pp_rank != 0:
                if last_iter:  # 最后一次迭代只需发送
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.isend,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=1,
                            msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                            stage="backward_step",
                            additional="send_prev",
                        )
                    )
                else:  # 非最后一次需要同时发送和接收
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.isend,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=1,
                            msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                            stage="backward_step",
                            additional="send_prev",
                        )
                    )
                    self.workload.append(
                        LogItem(
                            comm_type=CommType.irecv,
                            comm_group=CommGroup.pp_group,
                            comm_group_size=1,
                            msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                            stage="backward_step",
                            additional="recv_prev",
                        )
                    )

        # 处理预热阶段的反向传播通信
        for _ in range(pp_num_warmup_microbatches):
            if pp_rank != args.pipeline_model_parallel - 1:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.irecv,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="backward_step",
                        additional="recv_next",
                    )
                )
            # 反向传播计算
            self.workload.extend(self.model.backward())
            # 非首阶段需要发送梯度给前驱
            if pp_rank != 0:
                self.workload.append(
                    LogItem(
                        comm_type=CommType.isend,
                        comm_group=CommGroup.pp_group,
                        comm_group_size=1,
                        msg_size=2 * (args.hidden_size * args.seq_length * args.micro_batch),
                        stage="backward_step",
                        additional="send_prev",
                    )
                )

    def forward(self):
        """前向传播阶段的通信记录"""
        args = self.args
        # 张量并行组的参数广播
        if self.tp_is_enable:
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=5 * 8,  # 广播5个参数
                    stage="forward_step",
                    src=0,
                )
            )
            self.workload.append(
                LogItem(
                    comm_type=CommType.broadcast,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=8 * (args.world_size + args.seq_length * args.micro_batch),  # 输入数据大小
                    stage="forward_step",
                    src=0,
                )
            )
        # 前向传播计算
        self.workload.extend(self.model.forward())
        # 损失计算的跨数据并行组通信
        for _ in range(3):  # 三次all_reduce处理损失值
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.tp_group,
                    comm_group_size=self.args.tensor_model_parallel_size,
                    msg_size=args.micro_batch * args.seq_length * 4,  # float32类型大小为4字节
                    stage="forward_step._VocabParallelCrossEntropy",
                )
            )
        # 损失值的跨数据并行组平均
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.dp_group,
                comm_group_size=self.args.dp_num,
                msg_size=1 * 4,  # 单个float32数值
                stage="forward_step.average_losses_across_data_parallel_group",
            )
        )

    def backward(self):
        """反向传播阶段的通信记录"""
        self.workload.extend(self.model.backward())  # 添加模型反向传播的通信操作

    def step(self):
        """优化器步骤的通信记录"""
        args = self.args
        # 分布式优化器的梯度处理
        if args.use_distributed_optimizer:
            # reduce_scatter操作聚合梯度
            self.workload.append(
                LogItem(
                    comm_type=CommType.reduce_scatter,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=4 * self._get_total_params() // (args.pipeline_model_parallel),  # 梯度数据量
                    stage="step",
                )
            )
            # all_gather更新参数
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_gather,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=2 * self._get_total_params() // (args.pipeline_model_parallel),
                    stage="step",
                )
            )
        else:  # 非分布式优化器的梯度同步
            self.workload.append(
                LogItem(
                    comm_type=CommType.all_reduce,
                    comm_group=CommGroup.dp_group,
                    comm_group_size=self.args.dp_num,
                    msg_size=4 * self._get_total_params() // (args.pipeline_model_parallel),
                    stage="step.finish_grad_sync",
                )
            )
        # 层归一化梯度的跨张量并行组聚合
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.args.tensor_model_parallel_size,
                msg_size=2 * self._get_layernorm_params() // (args.pipeline_model_parallel),  # 层归一化参数量
                stage="step._allreduce_layernorm_grads",
            )
        )
        # 检查NaN值的通信
        self.workload.append(
            LogItem(
                comm_type=CommType.all_reduce,
                comm_group=CommGroup.tp_group,
                comm_group_size=self.args.tensor_model_parallel_size,
                msg_size=4,  # 单个float32数值
                stage="step.check_for_nan",
            )
        )


if __name__ == "__main__":
    args = get_params()  # 解析命令行参数
    model = MegatronModel(args)  # 创建Mock的Megatron模型
    workload_generator = MegatronWorkload(args, model)  # 创建工作负载生成器
    workload = workload_generator()  # 生成工作负载数据
    # 生成输出文件名
    filename = f"{workload_generator.name}_{args.model_name}_sp_{args.enable_sequence_parallel}_iteration_{args.epoch_num}_computationEnable_{args.computation_enable}_{args.world_size}n.csv"
    workload.dump(filename)  # 写入CSV文件
    # 可视化处理（如果启用）
    if args.enable_visual:
            try:
                from visualize.generate import visualize_output
                base_name = filename.split(".")[0]
                visualize_output(f"./results/mocked_workload/{base_name}_workload.csv",True)
            except ImportError: 
                print("visualize_output不可用，因为缺少依赖库")