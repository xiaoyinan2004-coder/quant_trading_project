# 量化交易系统 Quant Trading System

## 📊 项目简介

这是一个基于Python的量化交易系统，集成了数据获取、策略开发、回测验证和实盘交易功能。

## 🏗️ 项目结构

```
quant_trading_project/
├── data/                    # 数据存储
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后数据
│   └── cache/              # 缓存数据
├── strategies/             # 策略模块
│   ├── __init__.py
│   ├── base_strategy.py    # 基础策略类
│   ├── moving_average.py   # 双均线策略
│   ├── macd_strategy.py    # MACD策略
│   ├── rsi_strategy.py     # RSI策略
│   └── boll_strategy.py    # 布林带策略
├── indicators/             # 技术指标
│   ├── __init__.py
│   ├── moving_average.py   # 移动平均线
│   ├── macd.py            # MACD指标
│   ├── rsi.py             # RSI指标
│   ├── boll.py            # 布林带
│   └── kdj.py             # KDJ指标
├── backtest/               # 回测引擎
│   ├── __init__.py
│   ├── engine.py          # 回测引擎
│   ├── portfolio.py       # 投资组合
│   └── metrics.py         # 绩效指标
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── data_fetcher.py    # 数据获取
│   ├── logger.py          # 日志工具
│   └── config.py          # 配置管理
├── visualization/          # 可视化
│   ├── __init__.py
│   ├── charts.py          # 图表绘制
│   └── dashboard.py       # 仪表盘
├── api/                    # API接口
│   ├── __init__.py
│   └── broker_api.py      # 券商API
├── tests/                  # 测试文件
├── config/                 # 配置文件
│   ├── settings.yaml      # 主配置
│   └── strategies.yaml    # 策略配置
├── notebooks/              # Jupyter笔记
├── requirements.txt        # 依赖包
├── main.py                # 主程序入口
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd D:\quant_trading_project
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `config/settings.yaml` 文件，填入你的数据API密钥。

### 3. 运行示例策略

```bash
python main.py --strategy moving_average --symbol AAPL --start 2023-01-01 --end 2024-01-01
```

### 4. 启动回测

```bash
python -m backtest.engine --config config/strategies.yaml
```

## 📚 功能特性

- ✅ **数据获取**：支持Yahoo Finance、Tushare、AKShare等数据源
- ✅ **技术指标**：内置50+技术指标（MA、MACD、RSI、BOLL、KDJ等）
- ✅ **策略开发**：提供策略基类，支持快速开发新策略
- ✅ **回测引擎**：支持 Tick级和K线级回测，包含滑点和手续费模拟
- ✅ **绩效分析**：Sharpe比率、最大回撤、年化收益等指标
- ✅ **可视化**：Matplotlib/Plotly图表，交互式仪表盘
- ✅ **实盘交易**：支持多券商API对接

## 📝 开发计划

- [ ] 数据模块完成
- [ ] 技术指标库完成
- [ ] 回测引擎完成
- [ ] 基础策略实现
- [ ] 可视化界面
- [ ] 实盘交易接口
- [ ] 机器学习模块

## 👨‍💻 作者

基于B站量化交易课程(BV1bXCTBGE42)开发

## 📄 许可证

MIT License
