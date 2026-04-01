# 机器学习因子模型

项目中已新增基于 `XGBoost / LightGBM` 的横截面因子模型模块，适合做 A 股指数增强、选股打分和多因子排序。

## 新增能力

- `models/gradient_boosting_factor.py`
  - `PanelFactorDatasetBuilder`: 把多股票 OHLCV 数据转换为横截面因子面板
  - `GradientBoostingFactorModel`: 训练 `LightGBM` 或 `XGBoost` 回归模型，预测未来收益
  - `MachineLearningStockSelector`: 直接对原始股票数据打分并选出 Top-N
- `tests/test_gradient_boosting_factor.py`
  - 使用合成行情验证因子生成、模型训练、保存加载和选股流程

## 设计要点

- 复用现有 `factors/a_share_factors.py` 中的 A 股量价/技术面因子
- 自动生成未来 `label_horizon` 日收益标签
- 参考业界常见横截面因子建模流程，对每日因子做：
  - 截面 winsorize
  - 截面标准化
  - 时间顺序训练/验证切分
  - 训练集分位数裁剪和中位数缺失值填充
- 支持模型重要性分析、模型持久化、最新日期选股

## 使用示例

```python
from models.gradient_boosting_factor import (
    GradientBoostingFactorModel,
    MachineLearningStockSelector,
    PanelFactorDatasetBuilder,
)

# stock_data: {"000001": df1, "000002": df2, ...}
builder = PanelFactorDatasetBuilder()
panel = builder.build_dataset(stock_data, label_horizon=5)

model = GradientBoostingFactorModel(
    backend="lightgbm",
    model_params={
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 31,
    },
)
model.fit(panel, train_ratio=0.8)

selector = MachineLearningStockSelector(model, builder)
top_stocks = selector.select_stocks(stock_data, top_n=20)
print(top_stocks[["date", "symbol", "score"]].head())
print(model.feature_importance(top_n=10))
```

## 依赖

确保环境中安装以下包：

```bash
pip install scikit-learn xgboost lightgbm
```

## 参考思路

- `microsoft/qlib` 的 `LightGBM + Alpha158` 工作流
- `lightgbm-org/LightGBM` 官方仓库
- `dmlc/xgboost` 官方仓库
