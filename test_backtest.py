#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化回测测试脚本
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import setup_logger
from utils.data_fetcher import fetch_stock_data
from backtest.engine import BacktestEngine
from strategies.moving_average import MovingAverageStrategy

# 设置日志
logger = setup_logger()
logger.info("启动回测测试")

# 获取A股数据
symbol = '000001'  # 平安银行
start = '2023-01-01'
end = '2024-01-01'

try:
    logger.info(f"获取数据: {symbol}")
    data = fetch_stock_data(symbol, start, end, source='akshare')
    logger.info(f"获取到 {len(data)} 条数据")
    
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
    
    # 输出结果
    logger.info("=" * 50)
    logger.info("回测完成!")
    logger.info(f"初始资金: 100,000")
    logger.info(f"期末资金: {results['final_value']:,.2f}")
    logger.info(f"总收益率: {results['total_return']:.2%}")
    logger.info(f"年化收益率: {results['annual_return']:.2%}")
    logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
    logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")
    logger.info(f"交易次数: {results['total_trades']}")
    logger.info("=" * 50)
    
except Exception as e:
    logger.error(f"回测失败: {e}")
    import traceback
    traceback.print_exc()
