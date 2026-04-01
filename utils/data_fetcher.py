#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据获取模块
Data Fetcher Module

支持多种数据源：
- Yahoo Finance (yfinance) - 美股、港股、A股
- AKShare - A股数据
- Tushare - A股数据（需要API Key）
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from datetime import datetime, timedelta
import os

from .a_share_minute_data import AShareMinuteDataFetcher

# 尝试导入可选依赖
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class DataFetcher:
    """数据获取器"""
    
    def __init__(self, cache_dir: str = None):
        """
        初始化数据获取器
        
        Args:
            cache_dir: 数据缓存目录
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def fetch_yahoo(self, symbol: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
        """
        从Yahoo Finance获取数据
        
        Args:
            symbol: 股票代码 (如 'AAPL', 'MSFT', '0700.HK', '000001.SS')
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            interval: 时间周期 '1d', '1h', '1m'
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
        """
        if not YFINANCE_AVAILABLE:
            raise ImportError("请先安装 yfinance: pip install yfinance")
        
        print(f"[DataFetcher] 从Yahoo获取 {symbol} 数据...")
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        
        if df.empty:
            raise ValueError(f"未获取到数据: {symbol}")
        
        # 标准化列名
        df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # 移除不必要的列
        if 'dividends' in df.columns:
            df = df.drop('dividends', axis=1)
        if 'stock_splits' in df.columns:
            df = df.drop('stock_splits', axis=1)
            
        print(f"[DataFetcher] 获取成功: {len(df)} 条数据")
        return df
    
    def fetch_akshare_a_stock(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """
        从AKShare获取A股数据
        
        Args:
            symbol: A股代码 (如 '000001', '600000')
            start: 开始日期 'YYYYMMDD'
            end: 结束日期 'YYYYMMDD'
            
        Returns:
            DataFrame with columns: [open, high, low, close, volume]
        """
        if not AKSHARE_AVAILABLE:
            raise ImportError("请先安装 akshare: pip install akshare")
        
        print(f"[DataFetcher] 从AKShare获取 A股{symbol} 数据...")
        
        # 转换日期格式
        start = start.replace('-', '')
        end = end.replace('-', '')
        
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                                start_date=start, end_date=end, adjust="qfq")
        
        if df.empty:
            raise ValueError(f"未获取到数据: {symbol}")
        
        # 重命名列
        column_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }
        df = df.rename(columns=column_map)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        
        # 只保留需要的列
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        print(f"[DataFetcher] 获取成功: {len(df)} 条数据")
        return df

    def fetch_akshare_a_stock_minute(
        self,
        symbol: str,
        start_datetime: str,
        end_datetime: str,
        period: str = '1',
        adjust: str = '',
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """从 AKShare 获取 A 股分钟级历史数据。"""
        minute_fetcher = AShareMinuteDataFetcher(cache_dir=os.path.join(self.cache_dir, 'a_share_minute'))
        return minute_fetcher.fetch_historical(
            symbol=symbol,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            period=period,
            adjust=adjust,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )

    def fetch_akshare_a_stock_intraday(
        self,
        symbol: str,
        start_time: str = '09:15:00',
        end_time: str = '15:00:00',
        trade_date: Optional[str] = None,
        use_cache: bool = False,
        include_pre_market: bool = True,
    ) -> pd.DataFrame:
        """从 AKShare 获取当日分钟级盘前/盘中数据。"""
        minute_fetcher = AShareMinuteDataFetcher(cache_dir=os.path.join(self.cache_dir, 'a_share_minute'))
        return minute_fetcher.fetch_intraday(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            trade_date=trade_date,
            use_cache=use_cache,
            include_pre_market=include_pre_market,
        )
    
    def fetch_index(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """
        获取指数数据
        
        Args:
            index_code: 指数代码 (如 '000300' 沪深300, '000001' 上证指数)
            start: 开始日期 'YYYY-MM-DD'
            end: 结束日期 'YYYY-MM-DD'
            
        Returns:
            DataFrame
        """
        if AKSHARE_AVAILABLE:
            return self._fetch_akshare_index(index_code, start, end)
        else:
            raise ImportError("需要安装 akshare")
    
    def _fetch_akshare_index(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """使用AKShare获取指数数据"""
        start = start.replace('-', '')
        end = end.replace('-', '')
        
        df = ak.index_zh_a_hist(symbol=index_code, period="daily",
                                start_date=start, end_date=end)
        
        if df.empty:
            raise ValueError(f"未获取到指数数据: {index_code}")
        
        column_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }
        df = df.rename(columns=column_map)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        return df
    
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取A股股票列表
        
        Returns:
            DataFrame with columns: [code, name, industry]
        """
        if AKSHARE_AVAILABLE:
            df = ak.stock_zh_a_spot_em()
            return df[['代码', '名称', '所属行业']].rename(columns={
                '代码': 'code',
                '名称': 'name',
                '所属行业': 'industry'
            })
        else:
            raise ImportError("需要安装 akshare")
    
    def save_to_cache(self, df: pd.DataFrame, symbol: str, start: str, end: str):
        """
        保存数据到缓存
        
        Args:
            df: 数据DataFrame
            symbol: 股票代码
            start: 开始日期
            end: 结束日期
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start}_{end}.csv")
        df.to_csv(cache_file)
        print(f"[DataFetcher] 数据已缓存: {cache_file}")
    
    def load_from_cache(self, symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        Args:
            symbol: 股票代码
            start: 开始日期
            end: 结束日期
            
        Returns:
            DataFrame or None
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_{start}_{end}.csv")
        if os.path.exists(cache_file):
            print(f"[DataFetcher] 从缓存加载: {cache_file}")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return None


# 便捷函数
def fetch_stock_data(symbol: str, start: str, end: str, source: str = 'yahoo') -> pd.DataFrame:
    """
    获取股票数据的便捷函数
    
    Args:
        symbol: 股票代码
        start: 开始日期 'YYYY-MM-DD'
        end: 结束日期 'YYYY-MM-DD'
        source: 数据源 'yahoo' 或 'akshare'
        
    Returns:
        DataFrame
    """
    fetcher = DataFetcher()
    
    if source == 'yahoo':
        return fetcher.fetch_yahoo(symbol, start, end)
    elif source == 'akshare':
        return fetcher.fetch_akshare_a_stock(symbol, start, end)
    else:
        raise ValueError(f"不支持的数据源: {source}")
