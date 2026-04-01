#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACD指标计算模块
Moving Average Convergence Divergence

MACD是技术分析中最经典的趋势跟踪指标之一，由Gerald Appel于1970年代提出。
它通过计算两条不同周期的EMA（指数移动平均线）之差来判断价格趋势的变化。

计算公式：
- DIF（快线）= EMA(12) - EMA(26)
- DEA（慢线/信号线）= EMA(DIF, 9)
- MACD柱状图（柱状线）= 2 * (DIF - DEA)

参考：BV1bXCTBGE42 量化交易课程
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    计算指数移动平均线 (EMA)
    
    EMA公式：EMA_today = α * Price_today + (1-α) * EMA_yesterday
    其中 α = 2 / (period + 1)
    
    Args:
        series: 价格序列
        period: 周期
        
    Returns:
        EMA序列
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    计算MACD指标
    
    Args:
        close: 收盘价序列
        fast_period: 快线周期，默认12
        slow_period: 慢线周期，默认26
        signal_period: 信号线周期，默认9
        
    Returns:
        DataFrame包含：
        - dif: DIF快线
        - dea: DEA慢线（信号线）
        - macd: MACD柱状图（乘以2的传统算法）
        
    Example:
        >>> df = pd.DataFrame({'close': [10, 11, 12, 11, 13, 15, 14, 16, 18, 17]})
        >>> macd_df = calculate_macd(df['close'])
        >>> print(macd_df.tail())
    """
    # 计算快速和慢速EMA
    ema_fast = calculate_ema(close, fast_period)
    ema_slow = calculate_ema(close, slow_period)
    
    # DIF = 快线 - 慢线
    dif = ema_fast - ema_slow
    
    # DEA = DIF的EMA
    dea = calculate_ema(dif, signal_period)
    
    # MACD柱状图 = 2 * (DIF - DEA)
    macd = 2 * (dif - dea)
    
    return pd.DataFrame({
        'dif': dif,
        'dea': dea,
        'macd': macd
    })


def macd_signal(
    macd_df: pd.DataFrame,
    method: str = 'cross'
) -> pd.Series:
    """
    生成MACD交易信号
    
    Args:
        macd_df: MACD数据框（包含dif, dea, macd列）
        method: 信号生成方法
            - 'cross': DIF与DEA金叉/死叉
            - 'histogram': MACD柱状图由负转正/正转负
            - 'zero_line': DIF穿越零轴
            
    Returns:
        信号序列: 1(买入), -1(卖出), 0(持仓)
    """
    dif = macd_df['dif']
    dea = macd_df['dea']
    macd = macd_df['macd']
    
    signal = pd.Series(0, index=macd_df.index)
    
    if method == 'cross':
        # 金叉：DIF上穿DEA (前一日DIF<DEA，当日DIF>=DEA)
        golden_cross = (dif > dea) & (dif.shift(1) <= dea.shift(1))
        # 死叉：DIF下穿DEA (前一日DIF>DEA，当日DIF<=DEA)
        death_cross = (dif < dea) & (dif.shift(1) >= dea.shift(1))
        
        signal[golden_cross] = 1
        signal[death_cross] = -1
        
    elif method == 'histogram':
        # MACD柱状图由负转正（买入）
        pos_turn = (macd > 0) & (macd.shift(1) <= 0)
        # MACD柱状图由正转负（卖出）
        neg_turn = (macd < 0) & (macd.shift(1) >= 0)
        
        signal[pos_turn] = 1
        signal[neg_turn] = -1
        
    elif method == 'zero_line':
        # DIF上穿零轴（买入）
        above_zero = (dif > 0) & (dif.shift(1) <= 0)
        # DIF下穿零轴（卖出）
        below_zero = (dif < 0) & (dif.shift(1) >= 0)
        
        signal[above_zero] = 1
        signal[below_zero] = -1
    
    return signal


def macd_divergence(
    close: pd.Series,
    macd_df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    检测MACD背离信号
    
    顶背离：价格创新高，但MACD未创新高（看跌信号）
    底背离：价格创新低，但MACD未创新低（看涨信号）
    
    Args:
        close: 收盘价序列
        macd_df: MACD数据框
        lookback: 回望周期
        
    Returns:
        DataFrame包含背离信号
    """
    # 找局部高点和低点
    price_high = close.rolling(window=lookback, center=True).max() == close
    price_low = close.rolling(window=lookback, center=True).min() == close
    
    macd_high = macd_df['dif'].rolling(window=lookback, center=True).max() == macd_df['dif']
    macd_low = macd_df['dif'].rolling(window=lookback, center=True).min() == macd_df['dif']
    
    # 顶背离：价格新高，DIF未新高
    top_divergence = price_high & (~macd_high)
    # 底背离：价格新低，DIF未新低
    bottom_divergence = price_low & (~macd_low)
    
    return pd.DataFrame({
        'top_divergence': top_divergence,
        'bottom_divergence': bottom_divergence
    })


def add_macd_to_dataframe(
    df: pd.DataFrame,
    close_col: str = 'close',
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> pd.DataFrame:
    """
    将MACD指标添加到数据框
    
    Args:
        df: 原始数据框
        close_col: 收盘价列名
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        
    Returns:
        添加了MACD列的数据框
    """
    macd_df = calculate_macd(
        df[close_col],
        fast_period=fast_period,
        slow_period=slow_period,
        signal_period=signal_period
    )
    
    return pd.concat([df, macd_df], axis=1)


# 便捷函数
macd = calculate_macd


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    
    df = pd.DataFrame({'close': close}, index=dates)
    
    # 计算MACD
    macd_result = calculate_macd(df['close'])
    df = pd.concat([df, macd_result], axis=1)
    
    print("MACD计算结果：")
    print(df.tail(10))
    
    # 生成信号
    signals = macd_signal(macd_result)
    print("\n交易信号：")
    print(signals[signals != 0].tail(10))