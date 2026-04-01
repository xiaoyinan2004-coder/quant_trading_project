#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日志工具模块
Logger Module
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "quant_trading", level: int = logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Returns:
        配置好的Logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 控制台输出格式
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


class SimpleLogger:
    """简单日志类，用于快速打印"""
    
    @staticmethod
    def info(msg: str):
        print(f"[INFO] {msg}")
    
    @staticmethod
    def warning(msg: str):
        print(f"[WARNING] {msg}")
    
    @staticmethod
    def error(msg: str):
        print(f"[ERROR] {msg}")
    
    @staticmethod
    def debug(msg: str):
        print(f"[DEBUG] {msg}")
