#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
量化中性策略 (Alpha Hedging)
Market Neutral Strategy

核心逻辑：
- 多头：AI多因子选股组合
- 空头：做空对应股指期货 (IC中证500 / IM中证1000)
- 目标：剥离Beta，获取纯Alpha收益

注意：A股股指期货长期贴水，需要强大的选股能力覆盖对冲成本
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MarketNeutralStrategy:
    """
    市场中性策略
    
    通过做多股票组合 + 做空股指期货，实现市场中性
    """
    
    def __init__(self,
                 target_beta: float = 0.0,  # 目标Beta
                 max_exposure_pct: float = 0.02,  # 最大敞口
                 hedge_ratio_buffer: float = 0.05,  # 对冲比例缓冲
                 rebalance_freq: int = 5):  # 调仓频率（天）
        """
        初始化中性策略
        
        Args:
            target_beta: 目标Beta值（0=完全对冲）
            max_exposure_pct: 最大净敞口比例
            hedge_ratio_buffer: 对冲比例调整缓冲
            rebalance_freq: 调仓频率（交易日）
        """
        self.target_beta = target_beta
        self.max_exposure_pct = max_exposure_pct
        self.hedge_ratio_buffer = hedge_ratio_buffer
        self.rebalance_freq = rebalance_freq
        
        # 持仓
        self.long_positions = {}  # 多头持仓
        self.short_positions = {}  # 空头持仓（股指期货）
        
        # 记录
        self.last_rebalance = None
        self.trade_history = []
    
    def generate_signals(self,
                        stock_scores: Dict[str, float],
                        portfolio_value: float,
                        index_futures_price: float,
                        index_beta: float = 1.0) -> Dict:
        """
        生成交易信号
        
        Args:
            stock_scores: 股票得分 {code: score}
            portfolio_value: 组合市值
            index_futures_price: 股指期货价格
            index_beta: 指数Beta（通常=1）
            
        Returns:
            交易信号 {'long': [...], 'short': {...}}
        """
        signals = {
            'long': [],
            'short': {},
            'hedge_ratio': 0.0
        }
        
        # 1. 选股（多头）
        long_signals = self._select_long_stocks(stock_scores, portfolio_value)
        signals['long'] = long_signals
        
        # 2. 计算对冲比例
        long_value = sum(s['target_value'] for s in long_signals)
        hedge_ratio = self._calculate_hedge_ratio(long_value, index_beta)
        signals['hedge_ratio'] = hedge_ratio
        
        # 3. 股指期货对冲
        short_signal = self._calculate_futures_position(
            long_value, hedge_ratio, index_futures_price
        )
        signals['short'] = short_signal
        
        return signals
    
    def _select_long_stocks(self, 
                           stock_scores: Dict[str, float],
                           portfolio_value: float,
                           top_n: int = 50) -> List[Dict]:
        """
        选股逻辑
        
        选择得分最高的N只股票，等权配置
        """
        # 按得分排序
        sorted_stocks = sorted(stock_scores.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        selected = sorted_stocks[:top_n]
        
        # 等权配置
        weight = 1.0 / len(selected) if selected else 0
        target_value_per_stock = portfolio_value * weight
        
        signals = []
        for code, score in selected:
            signals.append({
                'code': code,
                'score': score,
                'weight': weight,
                'target_value': target_value_per_stock,
                'action': 'BUY'
            })
        
        return signals
    
    def _calculate_hedge_ratio(self, 
                              long_value: float,
                              index_beta: float) -> float:
        """
        计算对冲比例
        
        目标：使组合Beta接近target_beta
        """
        # 假设股票组合Beta约等于指数Beta（简化）
        portfolio_beta = index_beta
        
        # 需要的对冲比例
        # 如果target_beta=0，需要完全对冲
        # 如果target_beta=0.5，对冲一半
        hedge_ratio = (portfolio_beta - self.target_beta) / index_beta
        
        # 限制在合理范围
        hedge_ratio = max(0.5, min(1.2, hedge_ratio))
        
        return hedge_ratio
    
    def _calculate_futures_position(self,
                                   long_value: float,
                                   hedge_ratio: float,
                                   futures_price: float,
                                   contract_multiplier: int = 200) -> Dict:
        """
        计算股指期货头寸
        
        Args:
            contract_multiplier: 合约乘数（IC/IM=200）
            
        Returns:
            期货信号
        """
        # 需要对冲的名义价值
        hedge_value = long_value * hedge_ratio
        
        # 计算合约数量
        contracts = int(hedge_value / (futures_price * contract_multiplier))
        
        return {
            'instrument': 'IM',  # 中证1000期货
            'direction': 'SHORT',
            'contracts': contracts,
            'notional_value': contracts * futures_price * contract_multiplier,
            'hedge_ratio': hedge_ratio
        }
    
    def check_rebalance_needed(self, current_date: datetime) -> bool:
        """检查是否需要调仓"""
        if self.last_rebalance is None:
            return True
        
        days_since = (current_date - self.last_rebalance).days
        return days_since >= self.rebalance_freq
    
    def calculate_portfolio_beta(self, 
                                 stock_betas: Dict[str, float]) -> float:
        """
        计算组合Beta
        """
        if not self.long_positions:
            return 0.0
        
        total_value = sum(self.long_positions.values())
        weighted_beta = 0.0
        
        for code, value in self.long_positions.items():
            weight = value / total_value
            beta = stock_betas.get(code, 1.0)
            weighted_beta += weight * beta
        
        return weighted_beta
    
    def calculate_exposure(self) -> Dict:
        """
        计算当前敞口
        """
        long_value = sum(self.long_positions.values())
        short_value = sum(self.short_positions.values())
        
        net_exposure = long_value - short_value
        gross_exposure = long_value + short_value
        
        return {
            'long_value': long_value,
            'short_value': short_value,
            'net_exposure': net_exposure,
            'gross_exposure': gross_exposure,
            'net_exposure_pct': net_exposure / long_value if long_value > 0 else 0
        }
    
    def check_risk_limits(self) -> List[str]:
        """
        检查风险限制
        
        Returns:
            风险警告列表
        """
        warnings = []
        
        exposure = self.calculate_exposure()
        
        # 检查净敞口
        if abs(exposure['net_exposure_pct']) > self.max_exposure_pct:
            warnings.append(
                f"净敞口超限: {exposure['net_exposure_pct']:.2%} "
                f"> {self.max_exposure_pct:.2%}"
            )
        
        return warnings
    
    def estimate_costs(self,
                      long_turnover: float,
                      short_turnover: float,
                      holding_days: int = 20) -> Dict:
        """
        估算交易成本
        
        A股中性策略的主要成本：
        1. 股票交易费用
        2. 股指期货贴水成本
        3. 资金成本
        """
        # 股票交易费用（佣金+印花税）
        stock_commission = 0.00025  # 佣金
        stock_tax = 0.001  # 印花税（卖出）
        avg_trade_cost = stock_commission + stock_tax / 2  # 平均单边成本
        
        stock_trading_cost = (long_turnover * avg_trade_cost * 2 +  # 买卖
                             short_turnover * 0.00002 * 2)  # 期货
        
        # 股指期货贴水成本（年化约8-15%）
        futures_discount_rate = 0.10  # 假设10%贴水
        futures_cost = short_turnover * futures_discount_rate * (holding_days / 252)
        
        # 资金成本（融资融券利率约6-8%）
        funding_cost_rate = 0.07
        funding_cost = short_turnover * funding_cost_rate * (holding_days / 252)
        
        total_cost = stock_trading_cost + futures_cost + funding_cost
        
        return {
            'stock_trading_cost': stock_trading_cost,
            'futures_discount_cost': futures_cost,
            'funding_cost': funding_cost,
            'total_cost': total_cost,
            'cost_pct': total_cost / long_turnover if long_turnover > 0 else 0
        }
    
    def get_performance_metrics(self,
                               daily_returns: pd.Series,
                               benchmark_returns: pd.Series) -> Dict:
        """
        计算绩效指标
        """
        # 年化收益
        annual_return = daily_returns.mean() * 252
        
        # 波动率
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Beta
        covariance = daily_returns.cov(benchmark_returns)
        benchmark_variance = benchmark_returns.var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        # 与基准的相关性
        correlation = daily_returns.corr(benchmark_returns)
        
        # 最大回撤
        cumulative = (1 + daily_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'beta': beta,
            'correlation': correlation,
            'max_drawdown': max_drawdown,
            'annualized_alpha': annual_return - beta * benchmark_returns.mean() * 252
        }


class FuturesDataFetcher:
    """
    股指期货数据获取
    """
    
    # 合约代码
    FUTURES_CODES = {
        'IC': 'IC',      # 中证500期货
        'IM': 'IM',      # 中证1000期货
        'IF': 'IF',      # 沪深300期货
    }
    
    @staticmethod
    def get_futures_price(symbol: str = 'IM0') -> Optional[float]:
        """
        获取股指期货最新价格
        
        Args:
            symbol: 合约代码，IM0=中证1000主力合约
        """
        try:
            import akshare as ak
            
            # 获取期货行情
            df = ak.futures_zh_realtime(symbol=symbol)
            if not df.empty:
                return float(df['最新价'].iloc[0])
            return None
        except Exception as e:
            print(f"获取期货价格失败: {e}")
            return None
    
    @staticmethod
    def get_futures_info(symbol: str = 'IM') -> Dict:
        """
        获取期货合约信息
        """
        return {
            'symbol': symbol,
            'contract_multiplier': 200,  # 合约乘数
            'min_price_change': 0.2,     # 最小变动价位
            'margin_ratio': 0.12,        # 保证金比例
            'trading_hours': '09:30-11:30, 13:00-15:00'
        }


if __name__ == '__main__':
    print("市场中性策略测试...")
    
    # 模拟股票得分
    stock_scores = {
        '000001': 85.0,
        '000002': 78.0,
        '000333': 92.0,
        '000858': 75.0,
        '002415': 88.0,
        # ... 更多股票
    }
    
    # 初始化策略
    strategy = MarketNeutralStrategy(
        target_beta=0.0,  # 完全对冲
        max_exposure_pct=0.02
    )
    
    # 生成信号
    portfolio_value = 1000000  # 100万
    futures_price = 6500  # IM期货价格
    
    signals = strategy.generate_signals(
        stock_scores,
        portfolio_value,
        futures_price
    )
    
    print("\n=== 交易信号 ===")
    print(f"\n多头股票 ({len(signals['long'])}只):")
    for s in signals['long'][:5]:  # 只显示前5只
        print(f"  {s['code']}: 得分={s['score']:.1f}, 权重={s['weight']:.2%}")
    
    print(f"\n空头对冲:")
    print(f"  合约: {signals['short'].get('instrument')}")
    print(f"  方向: {signals['short'].get('direction')}")
    print(f"  手数: {signals['short'].get('contracts')}")
    print(f"  对冲比例: {signals['short'].get('hedge_ratio'):.2%}")
    
    # 估算成本
    costs = strategy.estimate_costs(
        long_turnover=portfolio_value,
        short_turnover=signals['short'].get('notional_value', 0),
        holding_days=20
    )
    
    print(f"\n=== 成本估算（20天持仓） ===")
    print(f"  股票交易成本: {costs['stock_trading_cost']:,.2f}")
    print(f"  期货贴水成本: {costs['futures_discount_cost']:,.2f}")
    print(f"  资金成本: {costs['funding_cost']:,.2f}")
    print(f"  总成本: {costs['total_cost']:,.2f} ({costs['cost_pct']:.2%})")
    print(f"\n  注意: 需要Alpha收益 > {costs['cost_pct']:.2%} 才能盈利")
