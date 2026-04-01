#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
策略测试脚本 - 测试所有策略
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
from strategies.macd_strategy import MACDStrategy
from strategies.rsi_strategy import RSIStrategy
from strategies.boll_strategy import BollStrategy

# 设置日志
logger = setup_logger()

def generate_mock_data(start_date='2023-01-01', days=252, trend='up'):
    """生成模拟股票数据"""
    dates = pd.date_range(start=start_date, periods=days, freq='B')
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, days)
    
    if trend == 'up':
        returns += 0.001
    elif trend == 'down':
        returns -= 0.001
    
    price = 100.0
    prices = []
    for r in returns:
        price *= (1 + r)
        prices.append(price)
    
    data = pd.DataFrame({
        'open': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
        'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
        'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
        'close': prices,
        'volume': np.random.randint(100000, 1000000, days)
    }, index=dates)
    
    return data

def test_strategy(strategy_class, strategy_name, params=None):
    """测试单个策略"""
    logger.info(f"\n{'='*60}")
    logger.info(f"测试策略: {strategy_name}")
    logger.info('='*60)
    
    try:
        symbol = 'MOCK'
        data = generate_mock_data('2023-01-01', 252, trend='up')
        
        strategy = strategy_class(params)
        
        engine = BacktestEngine(
            initial_capital=100000,
            commission_rate=0.0003,
            slippage=0.001
        )
        
        engine.initialize(strategy, symbol, data)
        results = engine.run()
        
        metrics = results.get('metrics', {})
        
        logger.info(f"✅ 回测完成!")
        logger.info(f"   期末资金: {results['final_value']:,.2f}")
        logger.info(f"   总收益率: {results['total_return']:.2%}")
        logger.info(f"   年化收益: {metrics.get('annual_return', 0):.2%}")
        logger.info(f"   最大回撤: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"   交易次数: {results['total_trades']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# 测试所有策略
strategies = [
    (MovingAverageStrategy, "双均线策略 (Moving Average)", None),
    (MACDStrategy, "MACD策略", None),
    (RSIStrategy, "RSI策略", None),
    (BollStrategy, "布林带策略", None),
]

logger.info("开始策略测试...")
results = []

for strategy_class, name, params in strategies:
    success = test_strategy(strategy_class, name, params)
    results.append((name, success))

# 总结
logger.info(f"\n{'='*60}")
logger.info("测试总结")
logger.info('='*60)
for name, success in results:
    status = "✅ 通过" if success else "❌ 失败"
    logger.info(f"{status}: {name}")

passed = sum(1 for _, s in results if s)
total = len(results)
logger.info(f"\n总计: {passed}/{total} 个策略通过测试")
