#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
布林带策略
Bollinger Bands Strategy

策略逻辑（均值回归）：
- 价格触及下轨时买入
- 价格触及上轨时卖出
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from indicators.boll import calculate_bollinger


class BollStrategy(BaseStrategy):
    """布林带策略"""
    
    def __init__(self, params=None):
        """
        初始化布林带策略
        
        Args:
            params: {
                'period': 计算周期，默认20
                'std_dev': 标准差倍数，默认2.0
            }
        """
        default_params = {
            'period': 20,
            'std_dev': 2.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def initialize(self, context):
        """初始化策略"""
        self.log_info(f"布林带策略初始化完成")
        self.log_info(f"周期: {self.params['period']}")
        self.log_info(f"标准差倍数: {self.params['std_dev']}")
        
    def handle_data(self, context, data):
        """处理数据"""
        symbol = context.symbol
        
        # 获取历史收盘价
        history = self.history(context, symbol, 'close', 
                              self.params['period'] + 20)
        
        if history is None or len(history) < self.params['period']:
            return
        
        # 计算布林带
        boll_df = calculate_bollinger(
            history,
            period=self.params['period'],
            std_dev=self.params['std_dev']
        )
        
        # 获取当前价格
        current_price = history.iloc[-1]
        prev_price = history.iloc[-2]
        
        # 获取布林带上下轨
        upper = boll_df['upper'].iloc[-1]
        lower = boll_df['lower'].iloc[-1]
        prev_upper = boll_df['upper'].iloc[-2]
        prev_lower = boll_df['lower'].iloc[-2]
        
        # 获取当前持仓
        current_position = context.get_position(symbol)
        
        # 均值回归策略
        # 价格从下轨反弹 - 买入信号
        buy_signal = (prev_price <= prev_lower) and (current_price > lower)
        # 价格从上轨回落 - 卖出信号
        sell_signal = (prev_price >= prev_upper) and (current_price < upper)
        
        # 交易逻辑
        if buy_signal and current_position == 0:
            # 触及下轨反弹买入
            self.log_info(f"🟢 布林带下轨反弹 - 买入 {symbol}")
            self.log_info(f"   当前价格: {current_price:.2f}")
            self.log_info(f"   下轨: {lower:.2f}")
            self.log_info(f"   上轨: {upper:.2f}")
            self.order_target_percent(context, symbol, 0.95)  # 95%仓位
            
        elif sell_signal and current_position > 0:
            # 触及上轨回落卖出
            self.log_info(f"🔴 布林带上轨回落 - 卖出 {symbol}")
            self.log_info(f"   当前价格: {current_price:.2f}")
            self.log_info(f"   上轨: {upper:.2f}")
            self.log_info(f"   下轨: {lower:.2f}")
            self.order_target_percent(context, symbol, 0)  # 清仓
    
    def before_trading_start(self, context):
        """交易开始前"""
        pass
    
    def after_trading_end(self, context):
        """交易结束后"""
        pass
