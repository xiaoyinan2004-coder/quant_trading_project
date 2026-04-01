#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
双均线策略
Moving Average Crossover Strategy

策略逻辑：
- 短期均线上穿长期均线（金叉）时买入
- 短期均线下穿长期均线（死叉）时卖出
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy


class MovingAverageStrategy(BaseStrategy):
    """双均线策略"""
    
    def __init__(self, params=None):
        """
        初始化双均线策略
        
        Args:
            params: {
                'short_window': 短期均线周期，默认5
                'long_window': 长期均线周期，默认20
            }
        """
        default_params = {
            'short_window': 5,
            'long_window': 20
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def initialize(self, context):
        """初始化策略"""
        self.log_info(f"双均线策略初始化完成")
        self.log_info(f"短期均线: {self.params['short_window']}日")
        self.log_info(f"长期均线: {self.params['long_window']}日")
        
        # 记录上一次的状态（用于判断金叉死叉）
        context.last_position = 0  # 0-空仓，1-持仓
        
    def handle_data(self, context, data):
        """处理数据"""
        symbol = context.symbol
        
        # 获取历史收盘价
        history = self.history(context, symbol, 'close', 
                              self.params['long_window'] + 10)
        
        if history is None or len(history) < self.params['long_window']:
            return
        
        # 计算均线
        short_ma = history.rolling(window=self.params['short_window']).mean().iloc[-1]
        long_ma = history.rolling(window=self.params['long_window']).mean().iloc[-1]
        
        # 获取上一天的均线（用于判断交叉）
        if len(history) >= self.params['long_window'] + 1:
            short_ma_prev = history.iloc[-2:-1].rolling(
                window=self.params['short_window']).mean().iloc[-1]
            long_ma_prev = history.iloc[-2:-1].rolling(
                window=self.params['long_window']).mean().iloc[-1]
        else:
            return
        
        # 获取当前持仓
        current_position = context.get_position(symbol)
        
        # 判断金叉死叉
        golden_cross = short_ma_prev <= long_ma_prev and short_ma > long_ma
        death_cross = short_ma_prev >= long_ma_prev and short_ma < long_ma
        
        # 交易逻辑
        if golden_cross and current_position == 0:
            # 金叉买入
            self.log_info(f"🟢 金叉信号 - 买入 {symbol}")
            self.log_info(f"   短期MA({self.params['short_window']}): {short_ma:.2f}")
            self.log_info(f"   长期MA({self.params['long_window']}): {long_ma:.2f}")
            self.order_target_percent(context, symbol, 0.95)  # 95%仓位
            
        elif death_cross and current_position > 0:
            # 死叉卖出
            self.log_info(f"🔴 死叉信号 - 卖出 {symbol}")
            self.log_info(f"   短期MA({self.params['short_window']}): {short_ma:.2f}")
            self.log_info(f"   长期MA({self.params['long_window']}): {long_ma:.2f}")
            self.order_target_percent(context, symbol, 0)  # 清仓
    
    def before_trading_start(self, context):
        """交易开始前"""
        pass
    
    def after_trading_end(self, context):
        """交易结束后"""
        pass


def calculate_ma(data, window):
    """
    计算移动平均线
    
    Args:
        data: 价格数据Series
        window: 窗口大小
        
    Returns:
        MA序列
    """
    return data.rolling(window=window).mean()


def detect_crossover(short_ma, long_ma):
    """
    检测均线交叉
    
    Args:
        short_ma: 短期均线序列
        long_ma: 长期均线序列
        
    Returns:
        1-金叉，-1-死叉，0-无交叉
    """
    if len(short_ma) < 2 or len(long_ma) < 2:
        return 0
    
    # 当前和前一天的位置关系
    curr_diff = short_ma.iloc[-1] - long_ma.iloc[-1]
    prev_diff = short_ma.iloc[-2] - long_ma.iloc[-2]
    
    if prev_diff <= 0 and curr_diff > 0:
        return 1  # 金叉
    elif prev_diff >= 0 and curr_diff < 0:
        return -1  # 死叉
    else:
        return 0  # 无交叉
