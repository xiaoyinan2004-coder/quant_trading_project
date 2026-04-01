# A 股分钟级数据模块

项目已新增 A 股分钟数据模块，核心文件是 `utils/a_share_minute_data.py`。

## 功能

- 从 AKShare 获取 A 股历史分钟级数据
- 获取当日盘前集合竞价 + 盘中分钟数据
- 标准化字段为 `open/high/low/close/volume/amount`
- 自动过滤非交易时段
- 本地 CSV 缓存
- 批量获取、多日拆分、分钟重采样

## 主要接口

```python
from utils.a_share_minute_data import AShareMinuteDataFetcher

fetcher = AShareMinuteDataFetcher()

# 历史分钟数据
df = fetcher.fetch_historical(
    symbol="000001",
    start_datetime="2026-03-31 09:30:00",
    end_datetime="2026-03-31 15:00:00",
    period="1",
    adjust="qfq",
)

# 当日盘前 + 盘中
intraday_df = fetcher.fetch_intraday(
    symbol="000001",
    start_time="09:15:00",
    end_time="15:00:00",
    include_pre_market=True,
)

# 重采样到 5 分钟
df_5m = fetcher.resample(df, rule="5min")
```

## 兼容旧入口

`utils/data_fetcher.py` 中也新增了两个代理方法：

- `DataFetcher.fetch_akshare_a_stock_minute(...)`
- `DataFetcher.fetch_akshare_a_stock_intraday(...)`

这样现有项目如果已经统一走 `DataFetcher`，可以直接复用。
