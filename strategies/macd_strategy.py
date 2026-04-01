#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACD策略
MACD Strategy

策略逻辑：
- DIF上穿DEA（金叉）时买入
- DIF下穿DEA（死叉）时卖出
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from indicators.macd import calculate_macd


class MACDStrategy(BaseStrategy):
    """MACD策略"""
    
    def __init__(self, params=None):
        """
        初始化MACD策略
        
        Args:
            params: {
                'fast_period': 快线周期，默认12
                'slow_period': 慢线周期，默认26
                'signal_period': 信号线周期，默认9
            }
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        
    def initialize(self, context):
        """初始化策略"""
        self.log_info(f"MACD策略初始化完成")
        self.log_info(f"快线周期: {self.params['fast_period']}")
        self.log_info(f"慢线周期: {self.params['slow_period']}")
        self.log_info(f"信号线周期: {self.params['signal_period']}")
        
    def handle_data(self, context, data):
        """处理数据"""
        symbol = context.symbol
        
        # 获取历史收盘价
        min_periods = self.params['slow_period'] + self.params['signal_period']
        history = self.history(context, symbol, 'close', min_periods + 20)
        
        if history is None or len(history) < min_periods:
            return
        
        # 计算MACD
        macd_df = calculate_macd(
            history,
            fast_period=self.params['fast_period'],
            slow_period=self.params['slow_period'],
            signal_period=self.params['signal_period']
        )
        
        # 获取当前和前一日的DIF、DEA
        dif = macd_df['dif']
        dea = macd_df['dea']
        
        current_dif = dif.iloc[-1]
        current_dea = dea.iloc[-1]
        prev_dif = dif.iloc[-2]
        prev_dea = dea.iloc[-2]
        
        # 获取当前持仓
        current_position = context.get_position(symbol)
        
        # 判断金叉死叉
        golden_cross = (prev_dif <= prev_dea) and (current_dif > current_dea)
        death_cross = (prev_dif >= prev_dea) and (current_dif < current_dea)
        
        # 交易逻辑
        if golden_cross and current_position == 0:
            # 金叉买入
            self.log_info(f"🟢 MACD金叉 - 买入 {symbol}")
            self.log_info(f"   DIF: {current_dif:.4f}")
            self.log_info(f"   DEA: {current_dea:.4f}")
            self.order_target_percent(context, symbol, 0.95)  # 95%仓位
            
        elif death_cross and current_position > 0:
            # 死叉卖出
            self.log_info(f"🔴 MACD死叉 - 卖出 {symbol}")
            self.log_info(f"   DIF: {current_dif:.4f}")
            self.log_info(f"   DEA: {current_dea:.4f}")
            self.order_target_percent(context, symbol, 0)  # 清仓
    
    def before_trading_start(self, context):
        """交易开始前"""
        pass
    
    def after_trading_end(self, context):
        """交易结束后"""
        pass
