import logging
import os
from typing import Dict, Optional, Union

from concurrent_log_handler import ConcurrentRotatingFileHandler


class ODLogger:
    """优化的日志记录器，专为odpath库设计

    特性：
    - 线程安全的日志轮转
    - 可自定义日志级别和格式
    - 同时输出到控制台和文件
    - 自动创建日志目录
    """

    # 预定义的日志级别映射
    LEVEL_MAP = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    # 默认配置
    DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s: %(message)s"

    DEFAULT_LOG_SIZE = 50 * 1024 * 1024  # 50MB
    DEFAULT_BACKUP_COUNT = 50

    def __init__(
        self,
        filename: Optional[str] = "odlog.log",
        level: Union[str, int] = "info",
        name: str = "odlog",
        log_format: Optional[str] = None,
        max_bytes: Optional[int] = None,
        backup_count: Optional[int] = None,
        log_dir: Optional[str] = None,
    ):
        """初始化日志记录器

        Args:
            name: 日志记录器名称
            filename: 日志文件名，如果为None则不记录到文件
            level: 日志级别('debug', 'info', 'warning', 'error', 'critical'或logging常量)
            log_format: 日志格式字符串
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份日志文件数量
            log_dir: 日志目录路径
        """
        self._logger = logging.getLogger(name)

        # 设置日志级别
        self.set_level(level)

        # 避免重复添加handler
        if not self._logger.handlers:
            # 创建格式化器
            formatter = logging.Formatter(log_format or self.DEFAULT_LOG_FORMAT)

            # 控制台handler
            self._add_stream_handler(formatter)

            # 文件handler（如果指定了文件名）
            if filename:
                self._add_file_handler(
                    filename=filename,
                    formatter=formatter,
                    max_bytes=max_bytes or self.DEFAULT_LOG_SIZE,
                    backup_count=backup_count or self.DEFAULT_BACKUP_COUNT,
                    log_dir=log_dir,
                )

    def _add_stream_handler(self, formatter: logging.Formatter):
        """添加控制台日志处理器"""
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        self._logger.addHandler(sh)

    def _add_file_handler(
        self,
        filename: str,
        formatter: logging.Formatter,
        max_bytes: int,
        backup_count: int,
        log_dir: Optional[str] = None,
    ):
        """添加文件日志处理器"""
        # 确保日志目录存在
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            filename = os.path.join(log_dir, filename)

        # 使用线程安全的日志轮转handler
        fh = ConcurrentRotatingFileHandler(
            filename=filename,
            mode="a",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setFormatter(formatter)
        self._logger.addHandler(fh)

    def set_level(self, level: Union[str, int]):
        """设置日志级别"""
        if isinstance(level, str):
            level = self.LEVEL_MAP.get(level.lower(), logging.INFO)
        self._logger.setLevel(level)

    def debug(self, msg: str, *args, **kwargs):
        """记录debug级别日志"""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """记录info级别日志"""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """记录warning级别日志"""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """记录error级别日志"""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """记录critical级别日志"""
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """记录异常日志"""
        self._logger.exception(msg, *args, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        """获取底层的logging.Logger对象"""
        return self._logger


# 模块级别的默认日志记录器
default_logger = ODLogger()


def get_logger(name: str = "odlog", **kwargs) -> ODLogger:
    """获取或创建一个ODLogger实例

    这是为了方便从库的任何地方访问相同的日志记录器。
    """
    return ODLogger(name=name, **kwargs)
