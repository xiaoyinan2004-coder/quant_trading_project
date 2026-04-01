#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
绩效指标计算模块
Performance Metrics Module

计算回测的绩效指标：
- 总收益率
- 年化收益率
- 最大回撤
- 夏普比率
- 索提诺比率
- 胜率
- 盈亏比
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime


def calculate_metrics(
    net_values: List[float],
    dates: List[datetime],
    initial_capital: float
) -> Dict[str, float]:
    """
    计算回测绩效指标
    
    Args:
        net_values: 每日净值列表
        dates: 日期列表
        initial_capital: 初始资金
        
    Returns:
        绩效指标字典
    """
    if len(net_values) < 2:
        return _empty_metrics()
    
    # 转换为numpy数组
    net_values = np.array(net_values)
    
    # 计算收益率序列
    returns = np.diff(net_values) / net_values[:-1]
    
    # 计算交易天数和年化因子
    total_days = len(net_values)
    trading_days_per_year = 252
    
    # 1. 总收益率
    total_return = (net_values[-1] - initial_capital) / initial_capital
    
    # 2. 年化收益率
    years = total_days / trading_days_per_year
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # 3. 最大回撤
    max_drawdown = calculate_max_drawdown(net_values)
    
    # 4. 夏普比率 (假设无风险利率为3%)
    risk_free_rate = 0.03
    if len(returns) > 0 and np.std(returns) > 0:
        sharpe_ratio = (np.mean(returns) * trading_days_per_year - risk_free_rate) / \
                       (np.std(returns) * np.sqrt(trading_days_per_year))
    else:
        sharpe_ratio = 0
    
    # 5. 索提诺比率 (只考虑下行波动)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and np.std(downside_returns) > 0:
        sortino_ratio = (np.mean(returns) * trading_days_per_year - risk_free_rate) / \
                        (np.std(downside_returns) * np.sqrt(trading_days_per_year))
    else:
        sortino_ratio = 0
    
    # 6. 波动率
    volatility = np.std(returns) * np.sqrt(trading_days_per_year) if len(returns) > 0 else 0
    
    # 7. 卡尔玛比率
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'volatility': volatility,
        'calmar_ratio': calmar_ratio,
        'total_days': total_days,
        'final_value': net_values[-1]
    }


def calculate_max_drawdown(net_values: np.ndarray) -> float:
    """
    计算最大回撤
    
    Args:
        net_values: 净值数组
        
    Returns:
        最大回撤比例
    """
    # 计算历史最高点
    peak = np.maximum.accumulate(net_values)
    
    # 计算回撤
    drawdown = (peak - net_values) / peak
    
    # 最大回撤
    max_drawdown = np.max(drawdown)
    
    return max_drawdown


def calculate_win_rate(trade_returns: List[float]) -> float:
    """
    计算胜率
    
    Args:
        trade_returns: 每笔交易的收益率列表
        
    Returns:
        胜率 (0-1)
    """
    if not trade_returns:
        return 0.0
    
    wins = sum(1 for r in trade_returns if r > 0)
    return wins / len(trade_returns)


def calculate_profit_loss_ratio(trade_returns: List[float]) -> float:
    """
    计算盈亏比
    
    Args:
        trade_returns: 每笔交易的收益率列表
        
    Returns:
        盈亏比
    """
    if not trade_returns:
        return 0.0
    
    avg_profit = np.mean([r for r in trade_returns if r > 0]) if any(r > 0 for r in trade_returns) else 0
    avg_loss = abs(np.mean([r for r in trade_returns if r < 0])) if any(r < 0 for r in trade_returns) else 1
    
    return avg_profit / avg_loss if avg_loss != 0 else 0


def _empty_metrics() -> Dict[str, float]:
    """返回空的绩效指标"""
    return {
        'total_return': 0.0,
        'annual_return': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0,
        'volatility': 0.0,
        'calmar_ratio': 0.0,
        'total_days': 0,
        'final_value': 0.0
    }


def format_metrics_report(metrics: Dict[str, float]) -> str:
    """
    格式化绩效报告
    
    Args:
        metrics: 绩效指标字典
        
    Returns:
        格式化后的报告字符串
    """
    report = """
╔══════════════════════════════════════════════════╗
║               回测绩效报告                        ║
╠══════════════════════════════════════════════════╣
║ 总收益率:        {total_return:12.2%}
║ 年化收益率:      {annual_return:12.2%}
║ 最大回撤:        {max_drawdown:12.2%}
║ 夏普比率:        {sharpe_ratio:12.2f}
║ 索提诺比率:      {sortino_ratio:12.2f}
║ 波动率:          {volatility:12.2%}
║ 卡尔玛比率:      {calmar_ratio:12.2f}
║ 交易天数:        {total_days:12d}
║ 期末资产:        {final_value:12,.2f}
╚══════════════════════════════════════════════════╝
""".format(**metrics)
    
    return report
