#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化交易系统主程序入口
Quant Trading System Main Entry
"""

import argparse
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logger
from utils.config import Config
from backtest.engine import BacktestEngine


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统')
    parser.add_argument('--mode', type=str, default='backtest', 
                       choices=['backtest', 'live', 'optimize'],
                       help='运行模式: backtest-回测, live-实盘, optimize-优化')
    parser.add_argument('--strategy', type=str, default='moving_average',
                       help='策略名称')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='股票代码')
    parser.add_argument('--start', type=str, default='2023-01-01',
                       help='开始日期')
    parser.add_argument('--end', type=str, default='2024-01-01',
                       help='结束日期')
    parser.add_argument('--capital', type=float, default=100000,
                       help='初始资金')
    parser.add_argument('--config', type=str, default='config/strategies.yaml',
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logger()
    logger.info(f"启动量化交易系统 - 模式: {args.mode}")
    
    # 加载配置
    config = Config(args.config)
    
    if args.mode == 'backtest':
        # 运行回测
        run_backtest(args, config, logger)
    elif args.mode == 'live':
        # 实盘交易
        run_live_trading(args, config, logger)
    elif args.mode == 'optimize':
        # 参数优化
        run_optimization(args, config, logger)


def run_backtest(args, config, logger):
    """运行回测"""
    logger.info(f"开始回测 - 策略: {args.strategy}, 标的: {args.symbol}")
    
    # 加载策略
    strategy_class = load_strategy(args.strategy)
    strategy = strategy_class()
    
    # 获取数据
    from utils.data_fetcher import fetch_stock_data
    try:
        data = fetch_stock_data(args.symbol, args.start, args.end, source='yahoo')
    except Exception as e:
        logger.error(f"获取数据失败: {e}")
        return
    
    # 初始化回测引擎
    engine = BacktestEngine(
        initial_capital=args.capital,
        start_date=args.start,
        end_date=args.end
    )
    
    # 初始化回测
    engine.initialize(strategy, args.symbol, data)
    
    # 运行回测
    results = engine.run()
    
    # 输出结果
    logger.info("回测完成!")
    logger.info(f"总收益率: {results['total_return']:.2%}")
    logger.info(f"年化收益: {results['annual_return']:.2%}")
    logger.info(f"最大回撤: {results['max_drawdown']:.2%}")
    logger.info(f"夏普比率: {results['sharpe_ratio']:.2f}")


def run_live_trading(args, config, logger):
    """运行实盘交易"""
    logger.info("启动实盘交易...")
    # TODO: 实现实盘交易逻辑
    pass


def run_optimization(args, config, logger):
    """运行参数优化"""
    logger.info("启动参数优化...")
    # TODO: 实现参数优化逻辑
    pass


def load_strategy(strategy_name):
    """加载策略类"""
    import importlib
    
    strategy_map = {
        'moving_average': 'strategies.moving_average.MovingAverageStrategy',
        'macd': 'strategies.macd_strategy.MACDStrategy',
        'rsi': 'strategies.rsi_strategy.RSIStrategy',
        'boll': 'strategies.boll_strategy.BollStrategy',
    }
    
    if strategy_name not in strategy_map:
        raise ValueError(f"未知策略: {strategy_name}")
    
    module_path, class_name = strategy_map[strategy_name].rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


if __name__ == '__main__':
    main()
