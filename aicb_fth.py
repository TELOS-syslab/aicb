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
import torch # 导入PyTorch库
from utils.utils import get_args, get_comp_out, extract_averages, Comp_with_aiob # 从utils模块导入工具函数
from utils.benchmark_logger import bench_logger # 导入benchmark_logger用于记录日志
from workload_generator.mocked_model.MockedDeepspeed import DeepspeedForCausalLM # 导入Deepspeed模型类
from workload_generator.mocked_model.MockedMegatron import MegatronModel # 导入Megatron模型类
from workload_generator.generate_deepspeed_stage1_2_workload import ( # 导入 DeepSpeed 第一和第二阶段生成器
    DeepSpeedStage1,
    DeepSpeedStage2,
)
from workload_generator.generate_deepspeed_stage3_workload import DeepSpeedStage3 # 导入DeepSpeed第三阶段生成器
from workload_generator.generate_megatron_workload import MegatronWorkload # 导入Megatron工作负载生成器
from workload_generator.generate_collective_test import Collective_Test # 导入集体测试生成器
from workload_applyer import WorkloadApplyer # 导入工作负载应用者
from utils.utils import * # 导入utils模块中的所有内容

if __name__ == "__main__": # 如果此模块是作为主程序执行
    args = get_args() # 获取命令行参数
    if not hasattr(args, "backend"): # 如果参数中不包含backend属性
        args.backend = "nccl" # 设置默认的分布式通信后端为nccl
    torch.distributed.init_process_group(backend=args.backend) # 初始化分布式进程组
    args.world_size = torch.distributed.get_world_size() # 获取总的进程数量，赋值给world_size
    args.rank = torch.distributed.get_rank() # 获取当前进程的排名
    if args.frame == "Megatron": # 如果选用的框架是Megatron
        model = MegatronModel(args) # 初始化Megatron模型
        workload_generator = MegatronWorkload(args, model) # 初始化Megatron工作负载生成器
    elif args.frame == "DeepSpeed": # 如果选用的框架是DeepSpeed
        model = DeepspeedForCausalLM(args) # 初始化DeepSpeed模型
        if args.stage == 1: # 如果使用的是DeepSpeed阶段1
            workload_generator = DeepSpeedStage1(args, model) # 使用DeepSpeed Stage1工作负载生成器
        elif args.stage == 2: # 如果使用的是DeepSpeed阶段2
            workload_generator = DeepSpeedStage2(args, model) # 使用DeepSpeed Stage2工作负载生成器
        elif args.stage == 3: # 如果使用的是DeepSpeed阶段3
            workload_generator = DeepSpeedStage3(args, model) # 使用DeepSpeed Stage3工作负载生成器
    elif args.frame == "collective_test": # 如果框架为集体测试
        workload_generator = Collective_Test(args, None) # 使用集体测试工作负载生成器
    workload = workload_generator() # 生成工作负载
    if args.aiob_enable and args.frame == "Megatron": # 如果启用了AIOB且框架是Megatron
        params = model.parameters() # 获取模型的参数
        args.model_param = sum(p.numel() for p in params) # 计算模型的参数数量
        if args.comp_filepath == None: # 如果计算文件路径未指定
            local_rank = torch.distributed.get_rank() % torch.cuda.device_count() # 计算本地排名
            if local_rank == 0:
                filepath = get_comp_out(args) # 获取计算输出路径
            else:
                filepath = get_aiob_path(args) # 获取AIOB路径
            torch.distributed.barrier() # 等待所有进程同步
            compute_cache = extract_averages(filepath,args) # 提取平均计算缓存
        else:
            print("comp_filepath:", args.comp_filepath) # 输出指定的计算文件路径
            compute_cache = extract_averages(args.comp_filepath,args) # 提取平均计算缓存
        workload = Comp_with_aiob(workload, compute_cache) # 用计算缓存更新工作负载
    if torch.distributed.get_rank() == 0: # 如果当前进程的排名为0
        filename = f"{workload_generator.name}_{args.model_name}_sp_{args.enable_sequence_parallel}_iteration_{args.epoch_num}_computationEnable_{args.computation_enable}_{args.world_size}n.csv" # 创建文件名
        workload.dump(filename) # 导出工作负载为CSV文件
    if not args.workload_only : # 如果不是仅生成工作负载
        applyer = WorkloadApplyer(workload=workload, args=args) # 初始化工作负载应用器
        cpu_time = applyer.apply_workload() # 应用工作负载并记录CPU时间
        if torch.distributed.get_rank() == 0: # 如果当前进程的排名为0
            bench_logger.analyze_comm_log() # 分析通信日志
            if args.frame != "collective_test": # 如果框架不是集体测试
                bench_logger.analyze_comm_time() # 分析通信时间
            csv_filename = bench_logger.dump_log(filename) # 导出日志为CSV文件
            if args.enable_visual: # 如果启用了可视化
                try:
                    from visualize.generate import visualize_output # 尝试导入可视化模块
                    visualize_output(csv_filename,False) # 生成可视化输出
                except ImportError: 
                    print("visualize_output is not available because required library is not found") # 提示未找到可视化所需的库

            print(
                f"total time for {args.frame} and {args.epoch_num} iterations is {cpu_time:.4f} s" # 输出总运行时间信息
            )