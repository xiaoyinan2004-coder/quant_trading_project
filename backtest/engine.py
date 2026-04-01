#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
回测引擎核心模块
Backtest Engine Core Module

提供完整的量化策略回测功能，包括：
- 账户管理（现金、持仓、市值）
- 订单执行（市价单、限价单）
- 绩效计算（收益率、回撤、夏普比率）
- 滑点和手续费模拟

参考：BV1bXCTBGE42 量化交易课程
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"      # 市价单
    LIMIT = "limit"        # 限价单
    STOP = "stop"          # 止损单


class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """订单对象"""
    symbol: str
    side: OrderSide
    amount: int
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    create_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    fill_price: Optional[float] = None
    commission: float = 0.0
    order_id: str = field(default_factory=lambda: f"ORD{np.random.randint(100000, 999999)}")


@dataclass
class Position:
    """持仓对象"""
    symbol: str
    amount: int = 0
    avg_cost: float = 0.0
    
    def market_value(self, current_price: float) -> float:
        """持仓市值"""
        return self.amount * current_price
    
    def profit_loss(self, current_price: float) -> float:
        """持仓盈亏"""
        return self.amount * (current_price - self.avg_cost)
    
    def profit_loss_pct(self, current_price: float) -> float:
        """持仓盈亏比例"""
        if self.avg_cost == 0:
            return 0.0
        return (current_price - self.avg_cost) / self.avg_cost


@dataclass
class Account:
    """账户对象"""
    initial_capital: float
    cash: float = field(init=False)
    positions: Dict[str, Position] = field(default_factory=dict)
    order_history: List[Order] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        self.cash = self.initial_capital
    
    def total_value(self, current_prices: Dict[str, float]) -> float:
        """账户总市值"""
        positions_value = sum(
            pos.market_value(current_prices.get(sym, 0))
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_value
    
    def get_position(self, symbol: str) -> Position:
        """获取持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]


class BacktestEngine:
    """
    回测引擎
    
    完整的回测系统，支持：
    - 多标的回测
    - 滑点和手续费设置
    - 多种订单类型
    - 详细的绩效报告
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,    # 手续费率 0.03%
        slippage: float = 0.001,             # 滑点 0.1%
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（默认万3）
            slippage: 滑点（默认千1）
            start_date: 回测开始日期
            end_date: 回测结束日期
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.start_date = start_date
        self.end_date = end_date
        
        # 账户
        self.account: Optional[Account] = None
        
        # 数据
        self.data: Optional[pd.DataFrame] = None
        self.current_date: Optional[datetime] = None
        self.current_bar: Optional[pd.Series] = None
        
        # 策略
        self.strategy: Optional[Any] = None
        self.symbol: Optional[str] = None
        
        # 记录
        self.daily_returns: List[Dict] = []
        self.net_values: List[float] = []
        self.dates: List[datetime] = []
        
    def initialize(self, strategy, symbol: str, data: pd.DataFrame):
        """
        初始化回测
        
        Args:
            strategy: 策略对象
            symbol: 交易标的
            data: 行情数据（需包含open/high/low/close/volume）
        """
        self.strategy = strategy
        self.symbol = symbol
        self.data = data.copy()
        
        # 初始化账户
        self.account = Account(initial_capital=self.initial_capital)
        
        # 初始化策略
        context = self._create_context()
        self.strategy.initialize(context)
        
        print(f"[回测引擎] 初始化完成")
        print(f"  初始资金: {self.initial_capital:,.2f}")
        print(f"  交易标的: {symbol}")
        print(f"  数据周期: {len(data)} 天")
        print(f"  手续费率: {self.commission_rate*100:.3f}%")
        print(f"  滑点: {self.slippage*100:.2f}%")
    
    def run(self) -> Dict[str, Any]:
        """
        运行回测
        
        Returns:
            回测结果字典
        """
        if self.data is None or self.strategy is None:
            raise ValueError("请先调用initialize()初始化回测")
        
        print(f"\n[回测引擎] 开始回测...")
        
        # 遍历每个交易日
        for idx, (date, bar) in enumerate(self.data.iterrows()):
            self.current_date = date
            self.current_bar = bar
            
            # 创建上下文
            context = self._create_context()
            
            # 调用策略的handle_data
            self.strategy.handle_data(context, bar)
            
            # 记录每日净值
            current_price = bar['close']
            total_value = self.account.total_value({self.symbol: current_price})
            self.net_values.append(total_value)
            self.dates.append(date)
            
            # 计算日收益率
            if idx > 0:
                daily_return = (self.net_values[-1] / self.net_values[-2]) - 1
            else:
                daily_return = 0
            self.daily_returns.append({
                'date': date,
                'return': daily_return,
                'net_value': total_value
            })
        
        print(f"[回测引擎] 回测完成\n")
        
        # 生成回测报告
        return self._generate_report()
    
    def _create_context(self):
        """创建策略上下文"""
        context = BacktestContext(
            engine=self,
            symbol=self.symbol,
            current_date=self.current_date,
            current_bar=self.current_bar
        )
        return context
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        from .metrics import calculate_metrics
        
        # 计算绩效指标
        metrics = calculate_metrics(
            net_values=self.net_values,
            dates=self.dates,
            initial_capital=self.initial_capital
        )
        
        # 最终持仓
        final_price = self.data['close'].iloc[-1]
        final_value = self.account.total_value({self.symbol: final_price})
        
        report = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': (final_value - self.initial_capital) / self.initial_capital,
            'total_trades': len(self.account.trade_history),
            'metrics': metrics,
            'net_values': pd.Series(self.net_values, index=self.dates),
            'daily_returns': pd.DataFrame(self.daily_returns),
            'trade_history': self.account.trade_history
        }
        
        return report


class BacktestContext:
    """回测上下文对象"""
    
    def __init__(self, engine: BacktestEngine, symbol: str, 
                 current_date: datetime, current_bar: pd.Series):
        self.engine = engine
        self.symbol = symbol
        self.current_date = current_date
        self.current_bar = current_bar
        self.account = engine.account
    
    @property
    def portfolio(self):
        """投资组合"""
        return self.account
    
    def order(self, symbol: str, amount: int):
        """下单"""
        if amount == 0:
            return
        
        side = OrderSide.BUY if amount > 0 else OrderSide.SELL
        order = Order(
            symbol=symbol,
            side=side,
            amount=abs(amount),
            create_time=self.current_date
        )
        
        self._execute_order(order)
    
    def order_target_percent(self, symbol: str, percent: float):
        """调仓至目标比例"""
        if not 0 <= percent <= 1:
            raise ValueError("比例必须在0-1之间")
        
        current_price = self.current_bar['close']
        total_value = self.account.total_value({symbol: current_price})
        target_value = total_value * percent
        
        current_position = self.account.get_position(symbol)
        current_value = current_position.amount * current_price
        
        delta_value = target_value - current_value
        amount = int(delta_value / current_price)
        
        if amount != 0:
            self.order(symbol, amount)
    
    def get_position(self, symbol: str) -> int:
        """获取持仓数量"""
        return self.account.get_position(symbol).amount
    
    def history(self, symbol: str, field: str, bar_count: int) -> pd.Series:
        """获取历史数据"""
        data = self.engine.data
        current_idx = data.index.get_loc(self.current_date)
        start_idx = max(0, current_idx - bar_count + 1)
        return data[field].iloc[start_idx:current_idx+1]
    
    def get_current_data(self, symbol: str) -> pd.Series:
        """获取当前数据"""
        return self.current_bar
    
    def _execute_order(self, order: Order):
        """执行订单"""
        current_price = self.current_bar['close']
        
        # 计算滑点
        if order.side == OrderSide.BUY:
            fill_price = current_price * (1 + self.engine.slippage)
        else:
            fill_price = current_price * (1 - self.engine.slippage)
        
        # 计算手续费
        trade_value = order.amount * fill_price
        commission = trade_value * self.engine.commission_rate
        
        # 更新订单
        order.fill_price = fill_price
        order.commission = commission
        order.fill_time = self.current_date
        order.status = OrderStatus.FILLED
        
        # 更新账户
        total_cost = trade_value + commission
        
        if order.side == OrderSide.BUY:
            if self.account.cash < total_cost:
                order.status = OrderStatus.REJECTED
                print(f"[订单拒绝] 资金不足: 需要{total_cost:.2f}, 可用{self.account.cash:.2f}")
                return
            
            self.account.cash -= total_cost
            position = self.account.get_position(order.symbol)
            # 更新平均成本
            total_amount = position.amount + order.amount
            if total_amount > 0:
                position.avg_cost = (position.amount * position.avg_cost + 
                                   order.amount * fill_price) / total_amount
            position.amount = total_amount
            
        else:  # SELL
            position = self.account.get_position(order.symbol)
            if position.amount < order.amount:
                order.status = OrderStatus.REJECTED
                print(f"[订单拒绝] 持仓不足: 需要{order.amount}, 可用{position.amount}")
                return
            
            self.account.cash += (trade_value - commission)
            position.amount -= order.amount
        
        # 记录订单和交易
        self.account.order_history.append(order)
        self.account.trade_history.append({
            'date': self.current_date,
            'symbol': order.symbol,
            'side': order.side.value,
            'amount': order.amount,
            'price': fill_price,
            'commission': commission,
            'total_value': trade_value
        })
        
        print(f"[订单成交] {order.side.value.upper()} {order.amount} {order.symbol} @ {fill_price:.2f}")
