from vidur.types.base_int_enum import BaseIntEnum


class EventType(BaseIntEnum):
    # at any given time step, call the schedule event at the last
    # 在任何给定的时间步，最后调用调度事件
    # to ensure that all the requests are processed
    # 以确保所有请求都被处理完毕
    BATCH_STAGE_ARRIVAL = 1
    # 批处理阶段到达
    REQUEST_ARRIVAL = 2
    # 请求到达
    BATCH_STAGE_END = 3
    # 批处理阶段结束
    BATCH_END = 4
    # 批处理结束
    GLOBAL_SCHEDULE = 5
    # 全局调度
    REPLICA_SCHEDULE = 6
    # 副本调度
    REPLICA_STAGE_SCHEDULE = 7
    # 副本阶段调度