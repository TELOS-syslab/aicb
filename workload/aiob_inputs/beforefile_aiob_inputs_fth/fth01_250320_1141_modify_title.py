import csv  # 导入csv模块

# def process_attention_file(file_path):  # 定义处理注意力文件的函数，参数为文件路径
#     with open(file_path, 'r') as file:  # 以读取模式打开指定路径的文件
#         reader = csv.reader(file)  # 创建CSV读取器
#         headers = next(reader)  # 读取CSV的标题行
#         data = next(reader)  # 读取CSV的第一条数据行
    
#     # 提取相关数据
#     attn_input_reshape_min = float(data[headers.index('time_stats.attn_input_reshape.min')])  # 获取'attn_input_reshape.min'的最小值并转换为浮点数
#     attn_input_reshape_max = float(data[headers.index('time_stats.attn_input_reshape.max')])  # 获取'attn_input_reshape.max'的最大值并转换为浮点数
#     attn_input_reshape_mean = float(data[headers.index('time_stats.attn_input_reshape.mean')])  # 获取'attn_input_reshape.mean'的平均值并转换为浮点数
    
#     attn_decode_min = float(data[headers.index('time_stats.attn_decode.min')])  # 获取'attn_decode.min'的最小值并转换为浮点数
#     attn_decode_max = float(data[headers.index('time_stats.attn_decode.max')])  # 获取'attn_decode.max'的最大值并转换为浮点数
#     attn_decode_mean = float(data[headers.index('time_stats.attn_decode.mean')])  # 获取'attn_decode.mean'的平均值并转换为浮点数
    
#     attn_output_reshape_min = float(data[headers.index('time_stats.attn_output_reshape.min')])  # 获取'attn_output_reshape.min'的最小值并转换为浮点数
#     attn_output_reshape_max = float(data[headers.index('time_stats.attn_output_reshape.max')])  # 获取'attn_output_reshape.max'的最大值并转换为浮点数
#     attn_output_reshape_mean = float(data[headers.index('time_stats.attn_output_reshape.mean')])  # 获取'attn_output_reshape.mean'的平均值并转换为浮点数
    
#     attn_prefill_min = float(data[headers.index('time_stats.attn_prefill.min')])  # 获取'attn_prefill.min'的最小值并转换为浮点数
#     attn_prefill_max = float(data[headers.index('time_stats.attn_prefill.max')])  # 获取'attn_prefill.max'的最大值并转换为浮点数
#     attn_prefill_mean = float(data[headers.index('time_stats.attn_prefill.mean')])  # 获取'attn_prefill.mean'的平均值并转换为浮点数
    
#     return {  # 返回一个包含处理后数据的字典
#         'Emb': {  # 'Emb'键对应的字典
#             'time_gpu_max': attn_input_reshape_max * 1000,  # 将最大时间转换为毫秒
#             'time_gpu_min': attn_input_reshape_min * 1000,  # 将最小时间转换为毫秒
#             'time_gpu_avg': attn_input_reshape_mean * 1000  # 将平均时间转换为毫秒
#         },
#         'layernorm': {  # 'layernorm'键对应的字典
#             'time_gpu_max': attn_decode_max * 1000,  # 将最大时间转换为毫秒
#             'time_gpu_min': attn_decode_min * 1000,  # 将最小时间转换为毫秒
#             'time_gpu_avg': attn_decode_mean * 1000  # 将平均时间转换为毫秒
#         },
#         'atten_qkv': {  # 'atten_qkv'键对应的字典
#             'time_gpu_max': attn_output_reshape_max * 1000,  # 将最大时间转换为毫秒
#             'time_gpu_min': attn_output_reshape_min * 1000,  # 将最小时间转换为毫秒
#             'time_gpu_avg': attn_output_reshape_mean * 1000  # 将平均时间转换为毫秒
#         },
#         'atten_flash': {  # 'atten_flash'键对应的字典
#             'time_gpu_max': attn_prefill_max * 1000,  # 将最大时间转换为毫秒
#             'time_gpu_min': attn_prefill_min * 1000,  # 将最小时间转换为毫秒
#             'time_gpu_avg': attn_prefill_mean * 1000  # 将平均时间转换为毫秒
#         }
#     }



def process_attention_file(file_path):  # 定义处理注意力文件的函数，参数为文件路径
    with open(file_path, 'r') as file:  # 以读取模式打开指定路径的文件
        reader = csv.reader(file)  # 创建CSV读取器
        headers = next(reader)  # 读取CSV的标题行
        data = next(reader)  # 读取CSV的第一条数据行
    
    # 提取相关数据，并处理空值
    def get_float_value(field_name, default=0.0):
        """从数据中获取字段值，若为空则返回默认值"""
        try:
            index = headers.index(field_name)  # 获取字段索引
            value = data[index]  # 获取字段值
            return float(value) if value.strip() else default  # 若为空字符串，则返回默认值
        except ValueError:
            return default  # 若转换失败，返回默认值
        except IndexError:
            return default  # 若字段不存在，返回默认值

    attn_input_reshape_min = get_float_value('time_stats.attn_input_reshape.min')  # 获取最小值
    attn_input_reshape_max = get_float_value('time_stats.attn_input_reshape.max')  # 获取最大值
    attn_input_reshape_mean = get_float_value('time_stats.attn_input_reshape.mean')  # 获取平均值
    
    attn_decode_min = get_float_value('time_stats.attn_decode.min')  # 获取最小值
    attn_decode_max = get_float_value('time_stats.attn_decode.max')  # 获取最大值
    attn_decode_mean = get_float_value('time_stats.attn_decode.mean')  # 获取平均值
    
    attn_output_reshape_min = get_float_value('time_stats.attn_output_reshape.min')  # 获取最小值
    attn_output_reshape_max = get_float_value('time_stats.attn_output_reshape.max')  # 获取最大值
    attn_output_reshape_mean = get_float_value('time_stats.attn_output_reshape.mean')  # 获取平均值
    
    attn_prefill_min = get_float_value('time_stats.attn_prefill.min')  # 获取最小值
    attn_prefill_max = get_float_value('time_stats.attn_prefill.max')  # 获取最大值
    attn_prefill_mean = get_float_value('time_stats.attn_prefill.mean')  # 获取平均值
    
    return {  # 返回一个包含处理后数据的字典
        'Emb': {  # 'Emb'键对应的字典 # fth 有问题attn_input_reshape_max 不是emb 处理一下
            'time_gpu_max': attn_input_reshape_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': attn_input_reshape_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': attn_input_reshape_mean * 1000  # 将平均时间转换为毫秒
        },
        'layernorm': {  # 'layernorm'键对应的字典
            'time_gpu_max': attn_decode_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': attn_decode_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': attn_decode_mean * 1000  # 将平均时间转换为毫秒
        },
        'atten_qkv': {  # 'atten_qkv'键对应的字典
            'time_gpu_max': attn_output_reshape_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': attn_output_reshape_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': attn_output_reshape_mean * 1000  # 将平均时间转换为毫秒
        },
        'atten_flash': {  # 'atten_flash'键对应的字典
            'time_gpu_max': attn_prefill_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': attn_prefill_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': attn_prefill_mean * 1000  # 将平均时间转换为毫秒
        }
    }
def process_mlp_file(file_path):  # 定义处理MLP文件的函数，参数为文件路径
    with open(file_path, 'r') as file:  # 以读取模式打开指定路径的文件
        reader = csv.reader(file)  # 创建CSV读取器
        headers = next(reader)  # 读取CSV的标题行
        data = next(reader)  # 读取CSV的第一条数据行
    
    # 提取相关数据
    emb_min = float(data[headers.index('time_stats.emb.min')])  # 获取'emb.min'的最小值并转换为浮点数
    emb_max = float(data[headers.index('time_stats.emb.max')])  # 获取'emb.max'的最大值并转换为浮点数
    emb_mean = float(data[headers.index('time_stats.emb.mean')])  # 获取'emb.mean'的平均值并转换为浮点数
    
    input_layernorm_min = float(data[headers.index('time_stats.input_layernorm.min')])  # 获取'input_layernorm.min'的最小值并转换为浮点数
    input_layernorm_max = float(data[headers.index('time_stats.input_layernorm.max')])  # 获取'input_layernorm.max'的最大值并转换为浮点数
    input_layernorm_mean = float(data[headers.index('time_stats.input_layernorm.mean')])  # 获取'input_layernorm.mean'的平均值并转换为浮点数
    
    attn_pre_proj_min = float(data[headers.index('time_stats.attn_pre_proj.min')])  # 获取'attn_pre_proj.min'的最小值并转换为浮点数
    attn_pre_proj_max = float(data[headers.index('time_stats.attn_pre_proj.max')])  # 获取'attn_pre_proj.max'的最大值并转换为浮点数
    attn_pre_proj_mean = float(data[headers.index('time_stats.attn_pre_proj.mean')])  # 获取'attn_pre_proj.mean'的平均值并转换为浮点数
    
    attn_rope_min = float(data[headers.index('time_stats.attn_rope.min')])  # 获取'attn_rope.min'的最小值并转换为浮点数
    attn_rope_max = float(data[headers.index('time_stats.attn_rope.max')])  # 获取'attn_rope.max'的最大值并转换为浮点数
    attn_rope_mean = float(data[headers.index('time_stats.attn_rope.mean')])  # 获取'attn_rope.mean'的平均值并转换为浮点数
    
    attn_post_proj_min = float(data[headers.index('time_stats.attn_post_proj.min')])  # 获取'attn_post_proj.min'的最小值并转换为浮点数
    attn_post_proj_max = float(data[headers.index('time_stats.attn_post_proj.max')])  # 获取'attn_post_proj.max'的最大值并转换为浮点数
    attn_post_proj_mean = float(data[headers.index('time_stats.attn_post_proj.mean')])  # 获取'attn_post_proj.mean'的平均值并转换为浮点数
    
    post_attention_layernorm_min = float(data[headers.index('time_stats.post_attention_layernorm.min')])  # 获取'post_attention_layernorm.min'的最小值并转换为浮点数
    post_attention_layernorm_max = float(data[headers.index('time_stats.post_attention_layernorm.max')])  # 获取'post_attention_layernorm.max'的最大值并转换为浮点数
    post_attention_layernorm_mean = float(data[headers.index('time_stats.post_attention_layernorm.mean')])  # 获取'post_attention_layernorm.mean'的平均值并转换为浮点数
    
    mlp_up_proj_min = float(data[headers.index('time_stats.mlp_up_proj.min')])  # 获取'mlp_up_proj.min'的最小值并转换为浮点数
    mlp_up_proj_max = float(data[headers.index('time_stats.mlp_up_proj.max')])  # 获取'mlp_up_proj.max'的最大值并转换为浮点数
    mlp_up_proj_mean = float(data[headers.index('time_stats.mlp_up_proj.mean')])  # 获取'mlp_up_proj.mean'的平均值并转换为浮点数
    
    mlp_act_min = float(data[headers.index('time_stats.mlp_act.min')])  # 获取'mlp_act.min'的最小值并转换为浮点数
    mlp_act_max = float(data[headers.index('time_stats.mlp_act.max')])  # 获取'mlp_act.max'的最大值并转换为浮点数
    mlp_act_mean = float(data[headers.index('time_stats.mlp_act.mean')])  # 获取'mlp_act.mean'的平均值并转换为浮点数
    
    mlp_down_proj_min = float(data[headers.index('time_stats.mlp_down_proj.min')])  # 获取'mlp_down_proj.min'的最小值并转换为浮点数
    mlp_down_proj_max = float(data[headers.index('time_stats.mlp_down_proj.max')])  # 获取'mlp_down_proj.max'的最大值并转换为浮点数
    mlp_down_proj_mean = float(data[headers.index('time_stats.mlp_down_proj.mean')])  # 获取'mlp_down_proj.mean'的平均值并转换为浮点数
    
    add_min = float(data[headers.index('time_stats.add.min')])  # 获取'add.min'的最小值并转换为浮点数
    add_max = float(data[headers.index('time_stats.add.max')])  # 获取'add.max'的最大值并转换为浮点数
    add_mean = float(data[headers.index('time_stats.add.mean')])  # 获取'add.mean'的平均值并转换为浮点数
    
    return {  # 返回一个包含处理后数据的字典
        'layernorm2': {  # 'layernorm2'键对应的字典
            'time_gpu_max': post_attention_layernorm_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': post_attention_layernorm_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': post_attention_layernorm_mean * 1000  # 将平均时间转换为毫秒
        },
        'mlp_linear_1': {  # 'mlp_linear_1'键对应的字典
            'time_gpu_max': mlp_up_proj_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': mlp_up_proj_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': mlp_up_proj_mean * 1000  # 将平均时间转换为毫秒
        },
        'mlp_gelu': {  # 'mlp_gelu'键对应的字典
            'time_gpu_max': mlp_act_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': mlp_act_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': mlp_act_mean * 1000  # 将平均时间转换为毫秒
        },
        'mlp_linear_2': {  # 'mlp_linear_2'键对应的字典
            'time_gpu_max': mlp_down_proj_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': mlp_down_proj_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': mlp_down_proj_mean * 1000  # 将平均时间转换为毫秒
        },
        'layernorm_post': {  # 'layernorm_post'键对应的字典
            'time_gpu_max': add_max * 1000,  # 将最大时间转换为毫秒
            'time_gpu_min': add_min * 1000,  # 将最小时间转换为毫秒
            'time_gpu_avg': add_mean * 1000  # 将平均时间转换为毫秒
        }
    }

def generate_output_file(attention_data, mlp_data, output_path):  # 定义生成输出文件的函数，参数为注意力数据、MLP数据和输出路径
    with open(output_path, 'w') as file:  # 以写入模式打开指定路径的文件
        file.write("train_iter:10\n")  # 写入训练迭代次数
        
        # 写入注意力数据
        for key, value in attention_data.items():  # 遍历注意力数据中的每个键值对
            file.write(f"{key}:\n")  # 写入键名
            file.write(f"    time_gpu_max: {value['time_gpu_max']:.0f}\n")  # 写入最大时间，格式为无小数点整数
            file.write(f"    time_gpu_min: {value['time_gpu_min']:.0f}\n")  # 写入最小时间，格式为无小数点整数
            file.write(f"    time_gpu_avg: {value['time_gpu_avg']:.0f}\n")  # 写入平均时间，格式为无小数点整数
        
        # 写入MLP数据
        for key, value in mlp_data.items():  # 遍历MLP数据中的每个键值对
            file.write(f"{key}:\n")  # 写入键名
            file.write(f"    time_gpu_max: {value['time_gpu_max']:.0f}\n")  # 写入最大时间，格式为无小数点整数
            file.write(f"    time_gpu_min: {value['time_gpu_min']:.0f}\n")  # 写入最小时间，格式为无小数点整数
            file.write(f"    time_gpu_avg: {value['time_gpu_avg']:.0f}\n")  # 写入平均时间，格式为无小数点整数

# 示例用法
# attention_file = "/disk1/futianhao/software1/aicb/vidur/data/profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv"  # 注意力CSV文件路径
attention_file = "vidur_profiling/compute/a100/meta-llama/Llama-2-7b-hf/attention.csv"  # 注意力CSV文件路径
mlp_file = "vidur_profiling/compute/a100/meta-llama/Llama-2-7b-hf/mlp.csv"  # MLP CSV文件路径
output_file = "Example_fth.txt"  # 输出文件路径

attention_data = process_attention_file(attention_file)  # 处理注意力文件并获取数据
mlp_data = process_mlp_file(mlp_file)  # 处理MLP文件并获取数据
generate_output_file(attention_data, mlp_data, output_file)  # 生成输出文件