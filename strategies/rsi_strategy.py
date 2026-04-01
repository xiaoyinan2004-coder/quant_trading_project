#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSI策略
RSI Strategy

策略逻辑：
- RSI < 30（超卖）时买入
- RSI > 70（超买）时卖出
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from indicators.rsi import calculate_rsi


class RSIStrategy(BaseStrategy):
    """RSI策略"""
    
    def __init__(self, params=None):
        """
        初始化RSI策略
        
        Args:
            params: {
                'period': RSI计算周期，默认14
                'oversold': 超卖阈值，默认30
                'overbought': 超买阈值，默认70
            }
        """
        default_params = {
            'period': 14,
            'oversold': 30,
            'overbought': 70
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def initialize(self, context):
        """初始化策略"""
        self.log_info(f"RSI策略初始化完成")
        self.log_info(f"RSI周期: {self.params['period']}")
        self.log_info(f"超卖阈值: {self.params['oversold']}")
        self.log_info(f"超买阈值: {self.params['overbought']}")
        
    def handle_data(self, context, data):
        """处理数据"""
        symbol = context.symbol
        
        # 获取历史收盘价
        history = self.history(context, symbol, 'close', 
                              self.params['period'] + 20)
        
        if history is None or len(history) < self.params['period']:
            return
        
        # 计算RSI
        rsi_values = calculate_rsi(history, period=self.params['period'])
        
        # 获取当前RSI
        current_rsi = rsi_values.iloc[-1]
        prev_rsi = rsi_values.iloc[-2]
        
        # 获取当前持仓
        current_position = context.get_position(symbol)
        
        oversold = self.params['oversold']
        overbought = self.params['overbought']
        
        # RSI从超卖区上穿 - 买入信号
        buy_signal = (prev_rsi <= oversold) and (current_rsi > oversold)
        # RSI从超买区下穿 - 卖出信号
        sell_signal = (prev_rsi >= overbought) and (current_rsi < overbought)
        
        # 交易逻辑
        if buy_signal and current_position == 0:
            # 超卖反弹买入
            self.log_info(f"🟢 RSI超卖反弹 - 买入 {symbol}")
            self.log_info(f"   RSI: {current_rsi:.2f} (超卖区上穿)")
            self.order_target_percent(context, symbol, 0.95)  # 95%仓位
            
        elif sell_signal and current_position > 0:
            # 超买回落卖出
            self.log_info(f"🔴 RSI超买回落 - 卖出 {symbol}")
            self.log_info(f"   RSI: {current_rsi:.2f} (超买区下穿)")
            self.order_target_percent(context, symbol, 0)  # 清仓
    
    def before_trading_start(self, context):
        """交易开始前"""
        pass
    
    def after_trading_end(self, context):
        """交易结束后"""
        pass
