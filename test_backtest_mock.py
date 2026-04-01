#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化回测测试脚本 - 使用模拟数据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.logger import setup_logger
from backtest.engine import BacktestEngine
from strategies.moving_average import MovingAverageStrategy

# 设置日志
logger = setup_logger()
logger.info("启动回测测试（模拟数据）")

# 生成模拟股票数据
def generate_mock_data(start_date='2023-01-01', days=252, trend='up'):
    """生成模拟股票数据"""
    dates = pd.date_range(start=start_date, periods=days, freq='B')  # 工作日
    
    # 生成价格序列（带趋势和噪声）
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)  # 日收益率
    
    if trend == 'up':
        returns += 0.001  # 添加上升趋势
    elif trend == 'down':
        returns -= 0.001  # 添加下降趋势
    
    # 计算价格
    price = 100.0
    prices = []
    for r in returns:
        price *= (1 + r)
        prices.append(price)
    
    # 生成OHLCV数据
    data = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days)
    }, index=dates)
    
    return data

try:
    # 生成模拟数据
    symbol = 'MOCK'
    data = generate_mock_data('2023-01-01', 252, trend='up')
    logger.info(f"生成模拟数据: {len(data)} 条")
    logger.info(f"数据范围: {data.index[0]} ~ {data.index[-1]}")
    logger.info(f"价格范围: {data['close'].min():.2f} ~ {data['close'].max():.2f}")
    
    # 创建策略
    strategy = MovingAverageStrategy(params={'short_window': 5, 'long_window': 20})
    
    # 创建回测引擎
    engine = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.0003,
        slippage=0.001
    )
    
    # 初始化回测
    engine.initialize(strategy, symbol, data)
    
    # 运行回测
    results = engine.run()
    
    # 获取metrics
    metrics = results.get('metrics', {})
    
    # 输出结果
    logger.info("=" * 50)
    logger.info("回测完成!")
    logger.info("=" * 50)
    logger.info(f"初始资金: {results['initial_capital']:,.0f}")
    logger.info(f"期末资金: {results['final_value']:,.2f}")
    logger.info(f"总收益率: {results['total_return']:.2%}")
    logger.info(f"年化收益率: {metrics.get('annual_return', 0):.2%}")
    logger.info(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}")
    logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"索提诺比率: {metrics.get('sortino_ratio', 0):.2f}")
    logger.info(f"波动率: {metrics.get('volatility', 0):.2%}")
    logger.info(f"卡尔玛比率: {metrics.get('calmar_ratio', 0):.2f}")
    logger.info(f"交易次数: {results['total_trades']}")
    logger.info("=" * 50)
    
    # 显示交易记录
    if results['trade_history']:
        logger.info("\n交易记录:")
        for trade in results['trade_history'][:5]:  # 只显示前5条
            logger.info(f"  {trade['date'].strftime('%Y-%m-%d')} {trade['side'].upper()} {trade['amount']}股 @ {trade['price']:.2f}")
    
except Exception as e:
    logger.error(f"回测失败: {e}")
    import traceback
    traceback.print_exc()
