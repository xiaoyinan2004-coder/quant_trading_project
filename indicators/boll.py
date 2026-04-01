#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
布林带指标 (Bollinger Bands)

布林带由John Bollinger于1980年代提出，是衡量价格波动性的重要工具。
它由三条线组成：中轨（移动平均线）、上轨和下轨。

计算公式：
- 中轨 (MB) = N日简单移动平均 (SMA)
- 上轨 (UP) = 中轨 + k × N日标准差
- 下轨 (DN) = 中轨 - k × N日标准差

常用参数：
- N = 20 (周期)
- k = 2 (标准差倍数)

交易策略：
- 价格触及上轨：可能超买，考虑卖出
- 价格触及下轨：可能超卖，考虑买入
- 带宽收窄：预示大行情即将发生
- 带宽扩张：趋势强劲

参考：BV1bXCTBGE42 量化交易课程
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


def calculate_bollinger(
    close: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    计算布林带指标
    
    Args:
        close: 收盘价序列
        period: 计算周期，默认20
        std_dev: 标准差倍数，默认2.0
        
    Returns:
        DataFrame包含：
        - middle: 中轨（移动平均线）
        - upper: 上轨
        - lower: 下轨
        - bandwidth: 带宽（(上轨-下轨)/中轨）
        - percent_b: %B指标（(价格-下轨)/(上轨-下轨)）
    """
    # 中轨：简单移动平均线
    middle = close.rolling(window=period).mean()
    
    # 标准差
    std = close.rolling(window=period).std()
    
    # 上轨和下轨
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    # 带宽（布林带的宽度，反映波动性）
    bandwidth = (upper - lower) / middle
    
    # %B指标（价格在布林带中的相对位置）
    percent_b = (close - lower) / (upper - lower)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': upper,
        'lower': lower,
        'bandwidth': bandwidth,
        'percent_b': percent_b
    })


def boll_signal(
    close: pd.Series,
    boll_df: pd.DataFrame,
    method: str = 'breakout'
) -> pd.Series:
    """
    生成布林带交易信号
    
    Args:
        close: 收盘价序列
        boll_df: 布林带数据框
        method: 信号方法
            - 'breakout': 突破上轨买入，突破下轨卖出（趋势跟踪）
            - 'reversal': 触及上轨卖出，触及下轨买入（均值回归）
            - 'squeeze': 带宽收窄后突破（ squeeze策略）
            
    Returns:
        信号序列: 1(买入), -1(卖出), 0(持仓)
    """
    upper = boll_df['upper']
    lower = boll_df['lower']
    middle = boll_df['middle']
    bandwidth = boll_df['bandwidth']
    
    signal = pd.Series(0, index=close.index)
    
    if method == 'reversal':
        # 均值回归策略
        # 触及上轨回落卖出
        sell = (close >= upper) & (close.shift(1) < upper.shift(1))
        # 触及下轨反弹买入
        buy = (close <= lower) & (close.shift(1) > lower.shift(1))
        
        signal[buy] = 1
        signal[sell] = -1
        
    elif method == 'breakout':
        # 趋势跟踪策略
        # 突破上轨买入
        buy = (close > upper) & (close.shift(1) <= upper.shift(1))
        # 跌破下轨卖出
        sell = (close < lower) & (close.shift(1) >= lower.shift(1))
        
        signal[buy] = 1
        signal[sell] = -1
        
    elif method == 'squeeze':
        # Squeeze策略：带宽收窄后突破
        # 计算带宽的移动平均
        bandwidth_ma = bandwidth.rolling(window=20).mean()
        # 带宽收窄
        squeeze = bandwidth < bandwidth_ma * 0.8
        # 带宽开始扩张且价格向上突破
        squeeze_breakout = (bandwidth > bandwidth.shift(1)) & squeeze.shift(1) & (close > upper.shift(1))
        # 带宽开始扩张且价格向下突破
        squeeze_breakdown = (bandwidth > bandwidth.shift(1)) & squeeze.shift(1) & (close < lower.shift(1))
        
        signal[squeeze_breakout] = 1
        signal[squeeze_breakdown] = -1
    
    return signal


def boll_squeeze(boll_df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """
    检测布林带收缩（Squeeze）
    
    Squeeze表示波动性降低，通常预示着大行情即将发生。
    
    Args:
        boll_df: 布林带数据框
        lookback: 回望周期
        
    Returns:
        Squeeze信号序列
    """
    bandwidth = boll_df['bandwidth']
    
    # 带宽处于历史低位
    bandwidth_low = bandwidth.rolling(window=lookback).min()
    squeeze = bandwidth <= bandwidth_low * 1.05  # 允许5%误差
    
    return squeeze


def boll_width(boll_df: pd.DataFrame) -> pd.Series:
    """
    计算布林带宽度变化率
    
    用于判断波动性的变化趋势。
    
    Args:
        boll_df: 布林带数据框
        
    Returns:
        带宽变化率
    """
    bandwidth = boll_df['bandwidth']
    return bandwidth.pct_change() * 100


def add_boll_to_dataframe(
    df: pd.DataFrame,
    close_col: str = 'close',
    period: int = 20,
    std_dev: float = 2.0
) -> pd.DataFrame:
    """
    将布林带指标添加到数据框
    
    Args:
        df: 原始数据框
        close_col: 收盘价列名
        period: 计算周期
        std_dev: 标准差倍数
        
    Returns:
        添加了布林带列的数据框
    """
    boll_df = calculate_bollinger(df[close_col], period=period, std_dev=std_dev)
    
    # 重命名列避免冲突
    boll_df = boll_df.rename(columns={
        'middle': f'boll_middle_{period}',
        'upper': f'boll_upper_{period}',
        'lower': f'boll_lower_{period}',
        'bandwidth': f'boll_width_{period}',
        'percent_b': f'boll_percent_b_{period}'
    })
    
    return pd.concat([df, boll_df], axis=1)


# 便捷函数
boll = calculate_bollinger


if __name__ == '__main__':
    # 测试
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    boll_df = calculate_bollinger(pd.Series(close, index=dates))
    signals = boll_signal(pd.Series(close, index=dates), boll_df, method='reversal')
    
    print("布林带统计:")
    print(boll_df.describe())
    print(f"\n交易信号:")
    print(f"  买入: {(signals == 1).sum()}")
    print(f"  卖出: {(signals == -1).sum()}")
