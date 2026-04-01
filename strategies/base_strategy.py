#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础策略类
Base Strategy Class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import pandas as pd


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, params: Dict = None):
        """
        初始化策略
        
        Args:
            params: 策略参数字典
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def initialize(self, context):
        """
        策略初始化
        
        Args:
            context: 上下文对象，包含账户、数据等信息
        """
        pass
    
    @abstractmethod
    def handle_data(self, context, data):
        """
        处理数据（每根K线调用一次）
        
        Args:
            context: 上下文对象
            data: 当前数据
        """
        pass
    
    def before_trading_start(self, context):
        """
        交易开始前调用（每天开盘前）
        
        Args:
            context: 上下文对象
        """
        pass
    
    def after_trading_end(self, context):
        """
        交易结束后调用（每天收盘后）
        
        Args:
            context: 上下文对象
        """
        pass
    
    def order_target_percent(self, context, symbol: str, percent: float):
        """
        调仓至目标持仓比例
        
        Args:
            context: 上下文对象
            symbol: 股票代码
            percent: 目标持仓比例(0-1)
        """
        if hasattr(context, 'order_target_percent'):
            context.order_target_percent(symbol, percent)
    
    def order(self, context, symbol: str, amount: int):
        """
        下单
        
        Args:
            context: 上下文对象
            symbol: 股票代码
            amount: 数量(正为买入，负为卖出)
        """
        if hasattr(context, 'order'):
            context.order(symbol, amount)
    
    def get_current_data(self, context, symbol: str) -> pd.Series:
        """
        获取当前数据
        
        Args:
            context: 上下文对象
            symbol: 股票代码
            
        Returns:
            当前数据Series
        """
        if hasattr(context, 'get_current_data'):
            return context.get_current_data(symbol)
        return None
    
    def history(self, context, symbol: str, field: str, bar_count: int) -> pd.Series:
        """
        获取历史数据
        
        Args:
            context: 上下文对象
            symbol: 股票代码
            field: 字段名(open/high/low/close/volume)
            bar_count: 数据条数
            
        Returns:
            历史数据Series
        """
        if hasattr(context, 'history'):
            return context.history(symbol, field, bar_count)
        return None
    
    def log_info(self, message: str):
        """输出信息日志"""
        print(f"[{self.name}] INFO: {message}")
    
    def log_warning(self, message: str):
        """输出警告日志"""
        print(f"[{self.name}] WARNING: {message}")
    
    def log_error(self, message: str):
        """输出错误日志"""
        print(f"[{self.name}] ERROR: {message}")
