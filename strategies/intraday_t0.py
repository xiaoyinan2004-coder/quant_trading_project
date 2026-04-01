#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股日内T+0交易策略
Intraday T+0 Trading Strategy for A-Share

利用已有底仓进行日内高抛低吸，增厚收益
核心逻辑：
1. 利用底仓，当日可卖出
2. 监控日内买卖点
3. 收盘前必须买回，避免隔夜敞口
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, time


class T0Strategy:
    """
    日内T+0策略
    
    适用于已有底仓的情况，通过日内做T增厚收益
    """
    
    def __init__(self, 
                 base_positions: Dict[str, int],
                 max_trades_per_day: int = 2,
                 stop_loss_pct: float = 0.005,
                 profit_target_pct: float = 0.01):
        """
        初始化T+0策略
        
        Args:
            base_positions: 底仓 {股票代码: 持仓数量}
            max_trades_per_day: 单股每日最大交易次数
            stop_loss_pct: 止损比例（如0.005=0.5%）
            profit_target_pct: 止盈比例（如0.01=1%）
        """
        self.base_positions = base_positions
        self.max_trades_per_day = max_trades_per_day
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        
        # 交易记录
        self.t0_positions = {}  # 当日买卖的临时仓位
        self.trade_history = []
        self.daily_trade_count = {code: 0 for code in base_positions}
    
    def reset_daily(self):
        """每日重置"""
        self.t0_positions = {}
        self.daily_trade_count = {code: 0 for code in self.base_positions}
    
    def generate_signals(self, 
                         tick_data: Dict[str, Dict],
                         current_time: datetime) -> List[Dict]:
        """
        生成交易信号
        
        Args:
            tick_data: 当前Tick数据 {code: {'price': x, 'volume': y, ...}}
            current_time: 当前时间
            
        Returns:
            交易信号列表
        """
        signals = []
        
        # 检查是否在交易时间
        if not self._is_trading_time(current_time):
            return signals
        
        for code, base_qty in self.base_positions.items():
            if code not in tick_data:
                continue
            
            # 检查当日交易次数
            if self.daily_trade_count.get(code, 0) >= self.max_trades_per_day:
                continue
            
            tick = tick_data[code]
            signal = self._analyze_single_stock(code, tick, base_qty)
            
            if signal:
                signals.append(signal)
        
        return signals
    
    def _analyze_single_stock(self, code: str, tick: Dict, base_qty: int) -> Optional[Dict]:
        """分析单只股票的买卖点"""
        current_price = tick['price']
        
        # 获取日内数据
        intraday_data = tick.get('intraday_bars', [])  # 分钟级数据
        if len(intraday_data) < 30:  # 需要足够的数据
            return None
        
        df = pd.DataFrame(intraday_data)
        
        # 计算日内指标
        vwap = self._calculate_vwap(df)  # 成交量加权平均价
        support = self._calculate_support(df)  # 支撑位
        resistance = self._calculate_resistance(df)  # 压力位
        
        # 当前持仓（考虑当日已做T的部分）
        current_hold = base_qty + self.t0_positions.get(code, 0)
        
        # 买卖逻辑
        signal = None
        
        # 买入信号：价格触及支撑位且有反弹迹象
        if current_price <= support * 1.002 and current_hold >= base_qty:
            # 确保不会超卖（T+1限制）
            if self.t0_positions.get(code, 0) > 0:
                # 已经买过，检查是否需要补仓
                avg_buy_price = self._get_avg_buy_price(code)
                if current_price < avg_buy_price * (1 - self.stop_loss_pct):
                    # 补仓摊低成本
                    signal = {
                        'code': code,
                        'action': 'BUY',
                        'price': current_price,
                        'quantity': min(100, base_qty),  # 买入100股或底仓的一部分
                        'reason': '支撑位补仓'
                    }
            else:
                signal = {
                    'code': code,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': min(100, base_qty),
                    'reason': '支撑位反弹'
                }
        
        # 卖出信号：价格触及压力位且已有持仓
        elif current_price >= resistance * 0.998 and current_hold > 0:
            if self.t0_positions.get(code, 0) >= 0:  # 没有超卖
                sell_qty = min(100, current_hold)  # 卖出100股或持仓的一部分
                signal = {
                    'code': code,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': sell_qty,
                    'reason': '压力位回落'
                }
        
        # VWAP偏离策略
        if not signal:
            signal = self._vwap_strategy(code, current_price, vwap, current_hold)
        
        # 更新临时仓位
        if signal:
            if signal['action'] == 'BUY':
                self.t0_positions[code] = self.t0_positions.get(code, 0) + signal['quantity']
            else:
                self.t0_positions[code] = self.t0_positions.get(code, 0) - signal['quantity']
            
            self.daily_trade_count[code] = self.daily_trade_count.get(code, 0) + 1
            self.trade_history.append({
                'time': datetime.now(),
                **signal
            })
        
        return signal
    
    def _vwap_strategy(self, code: str, price: float, vwap: float, hold_qty: int) -> Optional[Dict]:
        """VWAP偏离策略"""
        deviation = (price - vwap) / vwap
        
        # 价格低于VWAP 0.5%，买入
        if deviation < -0.005 and hold_qty >= self.base_positions.get(code, 0):
            return {
                'code': code,
                'action': 'BUY',
                'price': price,
                'quantity': min(100, self.base_positions.get(code, 0)),
                'reason': f'低于VWAP {deviation:.2%}'
            }
        
        # 价格高于VWAP 0.5%，卖出
        elif deviation > 0.005 and hold_qty > 0:
            return {
                'code': code,
                'action': 'SELL',
                'price': price,
                'quantity': min(100, hold_qty),
                'reason': f'高于VWAP {deviation:.2%}'
            }
        
        return None
    
    def _calculate_vwap(self, df: pd.DataFrame) -> float:
        """计算成交量加权平均价 (VWAP)"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).sum() / df['volume'].sum()
        return vwap
    
    def _calculate_support(self, df: pd.DataFrame) -> float:
        """计算日内支撑位"""
        # 简单方法：取日内低点和VWAP的均值
        day_low = df['low'].min()
        vwap = self._calculate_vwap(df)
        return (day_low + vwap) / 2
    
    def _calculate_resistance(self, df: pd.DataFrame) -> float:
        """计算日内压力位"""
        # 简单方法：取日内高点和VWAP的均值
        day_high = df['high'].max()
        vwap = self._calculate_vwap(df)
        return (day_high + vwap) / 2
    
    def _get_avg_buy_price(self, code: str) -> float:
        """获取平均买入价格"""
        buys = [t for t in self.trade_history 
                if t['code'] == code and t['action'] == 'BUY']
        if not buys:
            return 0
        total_cost = sum(t['price'] * t['quantity'] for t in buys)
        total_qty = sum(t['quantity'] for t in buys)
        return total_cost / total_qty if total_qty > 0 else 0
    
    def _is_trading_time(self, dt: datetime) -> bool:
        """检查是否在A股交易时间"""
        t = dt.time()
        # 上午 9:30-11:30
        morning_start = time(9, 30)
        morning_end = time(11, 30)
        # 下午 13:00-14:55 (14:55后不新开仓，准备平仓)
        afternoon_start = time(13, 0)
        afternoon_end = time(14, 55)
        
        return (
            (morning_start <= t <= morning_end) or
            (afternoon_start <= t <= afternoon_end)
        )
    
    def should_close_positions(self, dt: datetime) -> bool:
        """检查是否应该平仓（收盘前）"""
        return dt.time() >= time(14, 50)  # 14:50后必须平仓
    
    def generate_close_signals(self) -> List[Dict]:
        """
        生成平仓信号（收盘前买回/卖出，保持底仓不变）
        """
        signals = []
        
        for code, t0_qty in self.t0_positions.items():
            if t0_qty == 0:
                continue
            
            if t0_qty > 0:
                # 当日买多了，需要卖出
                signals.append({
                    'code': code,
                    'action': 'SELL',
                    'quantity': t0_qty,
                    'reason': '收盘前平仓'
                })
            else:
                # 当日卖多了，需要买回
                signals.append({
                    'code': code,
                    'action': 'BUY',
                    'quantity': abs(t0_qty),
                    'reason': '收盘前回补'
                })
        
        return signals
    
    def get_daily_summary(self) -> Dict:
        """获取当日交易汇总"""
        today_trades = [t for t in self.trade_history 
                       if t['time'].date() == datetime.now().date()]
        
        buys = [t for t in today_trades if t['action'] == 'BUY']
        sells = [t for t in today_trades if t['action'] == 'SELL']
        
        buy_amount = sum(t['price'] * t['quantity'] for t in buys)
        sell_amount = sum(t['price'] * t['quantity'] for t in sells)
        
        return {
            'date': datetime.now().date(),
            'total_trades': len(today_trades),
            'buy_trades': len(buys),
            'sell_trades': len(sells),
            'buy_amount': buy_amount,
            'sell_amount': sell_amount,
            'estimated_profit': sell_amount - buy_amount
        }


class T0Backtest:
    """T+0策略回测器"""
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 commission_rate: float = 0.00025):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
    
    def run_backtest(self, 
                     minute_data: Dict[str, pd.DataFrame],
                     base_positions: Dict[str, int]) -> Dict:
        """
        运行T+0回测
        
        Args:
            minute_data: 分钟级数据 {code: df}
            base_positions: 底仓
            
        Returns:
            回测结果
        """
        strategy = T0Strategy(base_positions)
        daily_pnl = []
        
        # 按日回测
        for date in self._get_dates(minute_data):
            strategy.reset_daily()
            day_trades = []
            
            # 获取当日分钟数据
            day_data = {code: df[df.index.date == date] 
                       for code, df in minute_data.items()}
            
            # 遍历每个分钟
            for timestamp in self._get_timestamps(day_data):
                tick_data = self._get_tick_data(day_data, timestamp)
                
                # 生成交易信号
                signals = strategy.generate_signals(tick_data, timestamp)
                day_trades.extend(signals)
                
                # 收盘前平仓
                if strategy.should_close_positions(timestamp):
                    close_signals = strategy.generate_close_signals()
                    day_trades.extend(close_signals)
            
            # 计算当日盈亏
            summary = strategy.get_daily_summary()
            daily_pnl.append(summary)
        
        return {
            'daily_pnl': daily_pnl,
            'total_trades': len(self.trades),
            'total_profit': sum(d['estimated_profit'] for d in daily_pnl)
        }
    
    def _get_dates(self, data: Dict[str, pd.DataFrame]) -> List:
        """获取所有日期"""
        dates = set()
        for df in data.values():
            dates.update(df.index.date)
        return sorted(list(dates))
    
    def _get_timestamps(self, day_data: Dict[str, pd.DataFrame]) -> List:
        """获取当日所有时间点"""
        timestamps = set()
        for df in day_data.values():
            timestamps.update(df.index)
        return sorted(list(timestamps))
    
    def _get_tick_data(self, day_data: Dict[str, pd.DataFrame], timestamp) -> Dict:
        """获取指定时间点的Tick数据"""
        tick_data = {}
        for code, df in day_data.items():
            if timestamp in df.index:
                row = df.loc[timestamp]
                tick_data[code] = {
                    'price': row['close'],
                    'volume': row['volume'],
                    'high': row['high'],
                    'low': row['low']
                }
        return tick_data


if __name__ == '__main__':
    print("T+0策略测试...")
    
    # 模拟底仓
    base_positions = {
        '000001': 1000,  # 平安银行1000股
        '000002': 500,   # 万科500股
    }
    
    strategy = T0Strategy(base_positions)
    
    # 模拟Tick数据
    tick_data = {
        '000001': {
            'price': 12.50,
            'volume': 50000,
            'intraday_bars': [
                {'high': 12.60, 'low': 12.40, 'close': 12.45, 'volume': 10000},
                {'high': 12.58, 'low': 12.42, 'close': 12.50, 'volume': 12000},
                # ... 更多分钟数据
            ]
        }
    }
    
    from datetime import datetime
    signals = strategy.generate_signals(tick_data, datetime.now())
    
    print(f"生成信号: {len(signals)}")
    for sig in signals:
        print(f"  {sig}")
