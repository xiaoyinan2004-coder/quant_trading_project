#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KDJ随机指标 (Stochastic Oscillator)

KDJ是期货和股票市场上最常用的技术分析工具之一，起源于随机指标K%D。
由George Lane于1950年代提出，后经改良加入J线，形成KDJ。

计算公式：
- RSV = (收盘价 - N日内最低价) / (N日内最高价 - N日内最低价) × 100
- K = 2/3 × 前一日K + 1/3 × 当日RSV
- D = 2/3 × 前一日D + 1/3 × 当日K
- J = 3K - 2D

参数：
- N: RSV周期，通常9
- M1: K线平滑周期，通常3
- M2: D线平滑周期，通常3

信号解读：
- K > D: 多头市场
- K < D: 空头市场
- J > 100: 超买
- J < 0: 超卖

参考：BV1bXCTBGE42 量化交易课程
"""

import pandas as pd
import numpy as np
from typing import Union, Optional


def calculate_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3
) -> pd.DataFrame:
    """
    计算KDJ指标
    
    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        n: RSV计算周期，默认9
        m1: K线平滑系数，默认3
        m2: D线平滑系数，默认3
        
    Returns:
        DataFrame包含：
        - k: K快线
        - d: D慢线
        - j: J线（3K-2D）
        - rsv: 未成熟随机值
    """
    # 计算N日内的最高最低价
    lowest_low = low.rolling(window=n).min()
    highest_high = high.rolling(window=n).max()
    
    # 计算RSV
    rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
    
    # 初始化K和D
    k = pd.Series(index=close.index, dtype=float)
    d = pd.Series(index=close.index, dtype=float)
    
    # 平滑计算K和D
    k.iloc[0] = 50
    d.iloc[0] = 50
    
    for i in range(1, len(close)):
        k.iloc[i] = (2/3) * k.iloc[i-1] + (1/3) * rsv.iloc[i]
        d.iloc[i] = (2/3) * d.iloc[i-1] + (1/3) * k.iloc[i]
    
    # 计算J线
    j = 3 * k - 2 * d
    
    return pd.DataFrame({
        'k': k,
        'd': d,
        'j': j,
        'rsv': rsv
    })


def kdj_signal(
    kdj_df: pd.DataFrame,
    method: str = 'cross'
) -> pd.Series:
    """
    生成KDJ交易信号
    
    Args:
        kdj_df: KDJ数据框
        method: 信号方法
            - 'cross': K线上穿/下穿D线（金叉死叉）
            - 'extreme': J线极端值（超买超卖）
            - 'combined': 综合信号
            
    Returns:
        信号序列: 1(买入), -1(卖出), 0(持仓)
    """
    k = kdj_df['k']
    d = kdj_df['d']
    j = kdj_df['j']
    
    signal = pd.Series(0, index=k.index)
    
    if method == 'cross':
        # 金叉：K上穿D
        golden_cross = (k > d) & (k.shift(1) <= d.shift(1))
        # 死叉：K下穿D
        death_cross = (k < d) & (k.shift(1) >= d.shift(1))
        
        signal[golden_cross] = 1
        signal[death_cross] = -1
        
    elif method == 'extreme':
        # J < 0 超卖买入
        oversold = (j < 0) & (j.shift(1) >= 0)
        # J > 100 超买卖出
        overbought = (j > 100) & (j.shift(1) <= 100)
        
        signal[oversold] = 1
        signal[overbought] = -1
        
    elif method == 'combined':
        # 综合信号：金叉且J < 20，或死叉且J > 80
        golden_cross = (k > d) & (k.shift(1) <= d.shift(1)) & (j < 20)
        death_cross = (k < d) & (k.shift(1) >= d.shift(1)) & (j > 80)
        
        signal[golden_cross] = 1
        signal[death_cross] = -1
    
    return signal


def kdj_status(kdj_df: pd.DataFrame) -> pd.Series:
    """
    判断KDJ状态
    
    Returns:
        状态标签：overbought(超买), oversold(超卖), neutral(中性)
    """
    j = kdj_df['j']
    
    status = pd.Series('neutral', index=j.index)
    status[j > 100] = 'overbought'
    status[j > 80] = 'high'
    status[j < 0] = 'oversold'
    status[j < 20] = 'low'
    
    return status


def add_kdj_to_dataframe(
    df: pd.DataFrame,
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    n: int = 9
) -> pd.DataFrame:
    """
    将KDJ指标添加到数据框
    
    Args:
        df: 原始数据框（需要包含high, low, close列）
        high_col: 最高价列名
        low_col: 最低价列名
        close_col: 收盘价列名
        n: RSV周期
        
    Returns:
        添加了KDJ列的数据框
    """
    kdj_df = calculate_kdj(
        df[high_col],
        df[low_col],
        df[close_col],
        n=n
    )
    return pd.concat([df, kdj_df], axis=1)


# 便捷函数
kdj = calculate_kdj


if __name__ == '__main__':
    # 测试
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    high = close + np.abs(np.random.randn(100)) * 2
    low = close - np.abs(np.random.randn(100)) * 2
    
    kdj_df = calculate_kdj(
        pd.Series(high, index=dates),
        pd.Series(low, index=dates),
        pd.Series(close, index=dates)
    )
    
    signals = kdj_signal(kdj_df)
    
    print("KDJ统计:")
    print(kdj_df.describe())
    print(f"\n交易信号:")
    print(f"  买入: {(signals == 1).sum()}")
    print(f"  卖出: {(signals == -1).sum()}")
