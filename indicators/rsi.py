#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSI相对强弱指标
Relative Strength Index

RSI是衡量价格变动速度和变化幅度的动量指标，由Welles Wilder于1978年提出。
取值范围0-100，通常用于判断超买超卖状态。

计算公式：
RSI = 100 - (100 / (1 + RS))
RS = 平均上涨幅度 / 平均下跌幅度

常用阈值：
- RSI > 70: 超买状态（可能回调）
- RSI < 30: 超卖状态（可能反弹）

参考：BV1bXCTBGE42 量化交易课程
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算RSI指标
    
    Args:
        close: 收盘价序列
        period: 计算周期，默认14
        
    Returns:
        RSI序列（0-100之间）
    """
    # 计算价格变化
    delta = close.diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 计算平均上涨和平均下跌（使用Wilder平滑方法）
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    # 计算RS
    rs = avg_gain / avg_loss
    
    # 计算RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def rsi_signal(
    rsi: pd.Series,
    overbought: float = 70,
    oversold: float = 30
) -> pd.Series:
    """
    生成RSI交易信号
    
    Args:
        rsi: RSI序列
        overbought: 超买阈值，默认70
        oversold: 超卖阈值，默认30
        
    Returns:
        信号序列: 1(买入), -1(卖出), 0(持仓)
        
    信号逻辑：
        - RSI从超卖区上穿：买入信号
        - RSI从超买区下穿：卖出信号
    """
    signal = pd.Series(0, index=rsi.index)
    
    # 超卖区上穿（买入）
    buy_signal = (rsi > oversold) & (rsi.shift(1) <= oversold)
    
    # 超买区下穿（卖出）
    sell_signal = (rsi < overbought) & (rsi.shift(1) >= overbought)
    
    signal[buy_signal] = 1
    signal[sell_signal] = -1
    
    return signal


def rsi_divergence(
    close: pd.Series,
    rsi: pd.Series,
    lookback: int = 20
) -> pd.DataFrame:
    """
    检测RSI背离
    
    顶背离：价格新高，RSI未新高（看跌）
    底背离：价格新低，RSI未新低（看涨）
    
    Args:
        close: 收盘价序列
        rsi: RSI序列
        lookback: 回望周期
        
    Returns:
        DataFrame包含背离信号
    """
    # 找局部极值点
    price_high = close.rolling(window=lookback, center=True).max() == close
    price_low = close.rolling(window=lookback, center=True).min() == close
    
    rsi_high = rsi.rolling(window=lookback, center=True).max() == rsi
    rsi_low = rsi.rolling(window=lookback, center=True).min() == rsi
    
    # 顶背离：价格新高，RSI未新高
    top_div = price_high & (~rsi_high)
    # 底背离：价格新低，RSI未新低
    bottom_div = price_low & (~rsi_low)
    
    return pd.DataFrame({
        'top_divergence': top_div,
        'bottom_divergence': bottom_div
    })


def rsi_strength_zone(rsi: pd.Series) -> pd.Series:
    """
    划分RSI强弱区域
    
    区域划分：
    - 80-100: 极强区（严重超买）
    - 50-80: 强势区
    - 50: 中性区
    - 20-50: 弱势区
    - 0-20: 极弱区（严重超卖）
    
    Args:
        rsi: RSI序列
        
    Returns:
        区域标签序列
    """
    zone = pd.Series('neutral', index=rsi.index)
    
    zone[rsi >= 80] = 'extreme_overbought'
    zone[(rsi >= 70) & (rsi < 80)] = 'overbought'
    zone[(rsi >= 50) & (rsi < 70)] = 'strong'
    zone[(rsi >= 30) & (rsi < 50)] = 'weak'
    zone[(rsi >= 20) & (rsi < 30)] = 'oversold'
    zone[rsi < 20] = 'extreme_oversold'
    
    return zone


def add_rsi_to_dataframe(
    df: pd.DataFrame,
    close_col: str = 'close',
    period: int = 14
) -> pd.DataFrame:
    """
    将RSI指标添加到数据框
    
    Args:
        df: 原始数据框
        close_col: 收盘价列名
        period: RSI周期
        
    Returns:
        添加了RSI列的数据框
    """
    rsi = calculate_rsi(df[close_col], period=period)
    df[f'rsi_{period}'] = rsi
    return df


# 便捷函数
rsi = calculate_rsi


if __name__ == '__main__':
    # 测试
    import numpy as np
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    rsi_values = calculate_rsi(pd.Series(close, index=dates))
    signals = rsi_signal(rsi_values)
    
    print("RSI统计:")
    print(f"  最大值: {rsi_values.max():.2f}")
    print(f"  最小值: {rsi_values.min():.2f}")
    print(f"  平均值: {rsi_values.mean():.2f}")
    print(f"\n交易信号数量:")
    print(f"  买入: {(signals == 1).sum()}")
    print(f"  卖出: {(signals == -1).sum()}")
