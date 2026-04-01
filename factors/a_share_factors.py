#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A股多因子计算模块
A-Share Factor Calculation Module

针对A股市场特征设计的因子：
- 量价因子：动量、波动率、成交量
- 技术因子：RSI、MACD、均线
- 情绪因子：涨停跌停、换手率
- 市值因子：小市值溢价
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class AShareFactorCalculator:
    """A股因子计算器"""
    
    def __init__(self):
        self.factor_names = []
    
    def calculate_all_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有因子
        
        Args:
            df: 包含OHLCV的数据框
            
        Returns:
            包含因子的数据框
        """
        factors = pd.DataFrame(index=df.index)
        
        # 基础价格数据
        factors['close'] = df['close']
        factors['volume'] = df['volume']
        
        # 1. 收益率因子
        factors = self._calculate_return_factors(df, factors)
        
        # 2. 波动率因子
        factors = self._calculate_volatility_factors(df, factors)
        
        # 3. 成交量因子
        factors = self._calculate_volume_factors(df, factors)
        
        # 4. 技术因子
        factors = self._calculate_technical_factors(df, factors)
        
        # 5. A股特有因子
        factors = self._calculate_ashare_specific_factors(df, factors)
        
        return factors
    
    def _calculate_return_factors(self, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """收益率因子"""
        close = df['close']
        
        # 不同周期的收益率
        factors['return_1d'] = close.pct_change(1)
        factors['return_5d'] = close.pct_change(5)
        factors['return_10d'] = close.pct_change(10)
        factors['return_20d'] = close.pct_change(20)
        
        # 收益率的偏度和峰度（A股散户多，分布非正态）
        factors['return_skew_20d'] = factors['return_1d'].rolling(20).skew()
        factors['return_kurt_20d'] = factors['return_1d'].rolling(20).kurt()
        
        # 动量因子（A股反转效应比动量效应更明显）
        factors['momentum_20d'] = close / close.shift(20) - 1
        factors['momentum_60d'] = close / close.shift(60) - 1
        
        return factors
    
    def _calculate_volatility_factors(self, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """波动率因子"""
        close = df['close']
        
        # 历史波动率
        factors['volatility_5d'] = close.pct_change().rolling(5).std() * np.sqrt(252)
        factors['volatility_20d'] = close.pct_change().rolling(20).std() * np.sqrt(252)
        factors['volatility_60d'] = close.pct_change().rolling(60).std() * np.sqrt(252)
        
        # 波动率的波动率
        factors['vol_of_vol'] = factors['volatility_20d'].rolling(20).std()
        
        # 振幅（A股散户多，振幅大往往伴随高波动）
        high, low = df['high'], df['low']
        factors['amplitude'] = (high - low) / close.shift(1)
        factors['amplitude_20d_avg'] = factors['amplitude'].rolling(20).mean()
        
        return factors
    
    def _calculate_volume_factors(self, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """成交量因子（A股量价关系很重要）"""
        volume = df['volume']
        close = df['close']
        
        # 成交量变化
        factors['volume_ratio'] = volume / volume.rolling(20).mean()
        factors['volume_change'] = volume.pct_change()
        
        # 成交额（更真实反映资金参与度）
        amount = close * volume
        factors['amount_ratio'] = amount / amount.rolling(20).mean()
        
        # 量价背离（价格上涨但成交量萎缩）
        factors['price_volume_divergence'] = (
            (close > close.shift(1)) & (volume < volume.shift(1))
        ).astype(int)
        
        return factors
    
    def _calculate_technical_factors(self, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """技术指标因子"""
        close = df['close']
        
        # RSI（A股超买超卖效果较好）
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        factors['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 均线位置（偏离度）
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()
        
        factors['price_to_ma5'] = close / ma5 - 1
        factors['price_to_ma20'] = close / ma20 - 1
        factors['price_to_ma60'] = close / ma60 - 1
        
        # 均线交叉
        factors['ma5_cross_ma20'] = ((ma5 > ma20) & (ma5.shift(1) <= ma20.shift(1))).astype(int)
        
        return factors
    
    def _calculate_ashare_specific_factors(self, df: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """A股特有因子"""
        close = df['close']
        high, low = df['high'], df['low']
        prev_close = close.shift(1)
        
        # 涨停/跌停因子（A股10%涨跌停限制，科创板/创业板20%）
        up_limit = prev_close * 1.1  # 主板涨停价
        down_limit = prev_close * 0.9  # 主板跌停价
        
        factors['is_up_limit'] = (close >= up_limit * 0.995).astype(int)  # 涨停
        factors['is_down_limit'] = (close <= down_limit * 1.005).astype(int)  # 跌停
        
        # 涨停打开（炸板）
        factors['limit_break'] = (
            (high >= up_limit * 0.995) & (close < high)
        ).astype(int)
        
        # 换手率（A股散户多，换手率高往往波动大）
        if 'turnover' in df.columns:
            factors['turnover'] = df['turnover']
        else:
            factors['turnover'] = np.nan
        
        # 开盘跳空（A股隔夜情绪影响大）
        open_price = df['open']
        factors['gap_up'] = (open_price > prev_close * 1.02).astype(int)  # 高开2%
        factors['gap_down'] = (open_price < prev_close * 0.98).astype(int)  # 低开2%
        
        # 日内走势（A股喜欢高开低走或低开高走）
        factors['intraday_trend'] = (close - open_price) / (high - low + 1e-8)
        
        return factors


class IndexEnhancementSelector:
    """
    指数增强选股器
    
    对标中证1000/国证2000，选取预期收益最高的股票
    """
    
    def __init__(self, model=None):
        self.model = model
        self.factor_calculator = AShareFactorCalculator()
    
    def select_stocks(self, 
                      stock_data: Dict[str, pd.DataFrame],
                      index_components: List[str],
                      top_n: int = 100) -> List[str]:
        """
        选股函数
        
        Args:
            stock_data: 所有股票的数据字典 {code: df}
            index_components: 指数成分股列表
            top_n: 选股数量
            
        Returns:
            选中的股票代码列表
        """
        scores = []
        
        for code in index_components:
            if code not in stock_data:
                continue
            
            df = stock_data[code]
            
            # 计算因子
            factors = self.factor_calculator.calculate_all_factors(df)
            
            # 获取最新因子值
            latest_factor_frame = factors.tail(1).copy()
            latest_factor_frame['date'] = pd.to_datetime([factors.index[-1]])
            latest_factor_frame['symbol'] = code
            latest_factors = latest_factor_frame.iloc[0]
            
            # 如果传入了机器学习模型，优先使用模型预测，否则回退到规则打分
            score = self._predict_model_score(latest_factor_frame)
            if score is None:
                score = self._calculate_score(latest_factors)
            
            scores.append({
                'code': code,
                'score': score,
                'factors': latest_factors
            })
        
        # 按得分排序
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 返回top_n股票
        return [s['code'] for s in scores[:top_n]]
    
    def _calculate_score(self, factors: pd.Series) -> float:
        """
        计算股票得分（简化版，实际使用机器学习模型）
        
        A股有效的简单逻辑：
        - 小市值（市值因子）
        - 低波动
        - 高换手（有资金关注）
        - 近期有动量但不过热
        """
        score = 0
        
        # 动量因子（20日涨幅适中的股票）
        if 'momentum_20d' in factors:
            mom = factors['momentum_20d']
            if 0.05 < mom < 0.3:  # 涨幅5%-30%
                score += 20
        
        # 波动率因子（偏好低波动）
        if 'volatility_20d' in factors:
            vol = factors['volatility_20d']
            if vol < 0.5:  # 年化波动率<50%
                score += 15
        
        # RSI因子（偏好超卖反弹）
        if 'rsi_14' in factors:
            rsi = factors['rsi_14']
            if 30 < rsi < 50:  # RSI在30-50之间
                score += 10
        
        # 成交量因子（有资金流入）
        if 'volume_ratio' in factors:
            vr = factors['volume_ratio']
            if vr > 1.5:  # 成交量放大
                score += 10
        
        # 避免涨停股（追高风险大）
        if 'is_up_limit' in factors and factors['is_up_limit'] == 1:
            score -= 50
        
        return score

    def _predict_model_score(self, latest_factor_frame: pd.DataFrame) -> Optional[float]:
        """Use a trained ML model when available."""
        if self.model is None or not hasattr(self.model, 'predict'):
            return None

        try:
            scored = self.model.predict(latest_factor_frame)
            if isinstance(scored, pd.DataFrame) and 'score' in scored.columns:
                return float(scored['score'].iloc[0])
        except Exception:
            return None

        return None


# 便捷函数
def get_index_components(index_code: str = '000852') -> List[str]:
    """
    获取指数成分股
    
    Args:
        index_code: 指数代码，000852=中证1000, 399303=国证2000
        
    Returns:
        成分股代码列表
    """
    try:
        import akshare as ak
        
        if index_code == '000852':  # 中证1000
            df = ak.index_stock_cons_weight_csindex(symbol="000852")
        elif index_code == '399303':  # 国证2000
            df = ak.index_stock_cons_weight_csindex(symbol="399303")
        else:
            # 默认返回中证1000
            df = ak.index_stock_cons_weight_csindex(symbol="000852")
        
        return df['成分券代码'].tolist()
    except Exception as e:
        print(f"获取指数成分股失败: {e}")
        return []


if __name__ == '__main__':
    # 测试
    print("A股因子计算器测试...")
    
    # 生成模拟数据
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    np.random.seed(42)
    
    mock_data = pd.DataFrame({
        'open': 10 + np.cumsum(np.random.randn(100) * 0.1),
        'high': 10 + np.cumsum(np.random.randn(100) * 0.1) + 0.1,
        'low': 10 + np.cumsum(np.random.randn(100) * 0.1) - 0.1,
        'close': 10 + np.cumsum(np.random.randn(100) * 0.1),
        'volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    calculator = AShareFactorCalculator()
    factors = calculator.calculate_all_factors(mock_data)
    
    print(f"计算了 {len(factors.columns)} 个因子")
    print(f"\n因子列表: {list(factors.columns)}")
    print(f"\n最新因子值:\n{factors.iloc[-1]}")
