# Adapted from
# https://github.com/skypilot-org/skypilot/blob/86dc0f6283a335e4aa37b3c10716f90999f48ab6/sky/sky_logging.py
"""Logging configuration for Sarathi."""
# 从logging模块导入必要的类和模块
import logging
import sys

# 定义日志格式
_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"  # 日志的输出格式，包含日志级别、时间戳、文件名、行号和消息
# 定义日期格式
_DATE_FORMAT = "%m-%d %H:%M:%S"  # 日志记录中的日期格式

# 定义自定义的日志格式化器类
class NewLineFormatter(logging.Formatter):  # 创建一个继承自logging.Formatter的类，用于格式化日志消息
    """Adds logging prefix to newlines to align multi-line messages."""  # 在多行消息中添加日志前缀以对齐

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)  # 调用父类的构造函数，传递格式和日期格式

    def format(self, record):
        msg = logging.Formatter.format(self, record)  # 获取原始格式化消息
        if record.message != "":  # 检查消息不为空
            parts = msg.split(record.message)  # 将消息内容从格式化字符串中分隔出来
            msg = msg.replace("\n", "\r\n" + parts[0])  # 在换行符之前插入前缀
        return msg  # 返回格式化后的消息

# 创建一个命名为“vidur”的根日志记录器
_root_logger = logging.getLogger("vidur")  # 获取或创建一个名为'vidur'的logger对象
_default_handler = None  # 初始化默认的处理器为空

# 定义设置日志记录器的方法
def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)  # 设置日志记录级别为DEBUG，记录所有级别的日志（DEBUG及以上）
    global _default_handler  # 使用全局变量_default_handler
    if _default_handler is None:  # 如果默认的处理器未被设置
        _default_handler = logging.StreamHandler(sys.stdout)  # 创建一个将日志输出到标准输出的StreamHandler
        _default_handler.flush = sys.stdout.flush  # type: ignore  # 将处理器的flush方法设置为标准输出的flush方法
        _default_handler.setLevel(logging.INFO)  # 设置处理器的日志级别为INFO
        _root_logger.addHandler(_default_handler)  # 将处理器添加到_root_logger中
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)  # 创建使用自定义格式化器的格式对象
    _default_handler.setFormatter(fmt)  # 将格式对象设置为处理器的格式
    _root_logger.propagate = False  # 防止日志消息传播到父日志记录器

# 日志记录器在模块导入时初始化
# 由于Python的全局解释器锁（GIL）的保证，这个操作是线程安全的，模块只会被导入一次
_setup_logger()  # 调用_setup_logger函数初始化日志记录器

# 定义初始化特定名称日志记录器的方法
def init_logger(name: str):
    return logging.getLogger(name)  # 返回一个特定名称的日志记录器对象