# M3-Net Project Architecture

## 1. 文档目标

这份文档定义 `quant_trading_project` 后续演进的目标架构：

- 以 `M3-Net (Multimodal-MoE-Memory Network)` 作为中长期核心研究框架
- 保留现有 `LightGBM / XGBoost` 作为强基线与控制组
- 让日频多因子、分钟级时序、文本信息、图结构关系和执行策略最终进入同一个可扩展系统

这不是一版“立刻全量实现”的设计，而是：

1. 先让项目具备稳健基线
2. 再逐层引入复杂模型
3. 保证每一层都能单独回测、对照、替换

## 2. 总体定位

`M3-Net` 在本项目中的角色，不是单一预测器，而是一个多层协同架构：

- `MLLM` 负责金融文本与多模态语义理解
- `MoE` 负责专家化分工与动态路由
- `World Model` 负责市场状态建模、情景模拟和 latent rollout
- `Causal ML` 负责减少伪相关、加强干预分析和稳健验证
- `Neuro-Symbolic` 负责把学习系统与显式交易规则、风控约束、业务知识图谱连接

项目最终目标不是“预测涨跌”这么简单，而是形成一个完整闭环：

`数据接入 -> 表征学习 -> 信号生成 -> 组合构建 -> 执行决策 -> 回测评估 -> 在线更新`

## 3. 适用问题

M3-Net 主要服务以下三类任务：

### 3.1 日频横截面选股

- 指数增强
- 多因子选股
- 风格轮动
- 中性组合打分

### 3.2 分钟级时序与执行

- 分钟级 alpha
- 日内 T0
- 交易执行优化
- 仓位/下单强度控制

### 3.3 研究辅助与语义增强

- 新闻/公告/研报理解
- 事件驱动信号
- 金融问答与因子解释
- 策略可解释性输出

## 4. 架构总览

建议把 M3-Net 分成七层：

1. 数据层
2. 特征与表示层
3. 记忆层
4. 专家路由层
5. 决策层
6. 组合与执行层
7. 评估与治理层

### 4.1 数据层

输入源分为五类：

- 市场行情：日频、分钟频、盘口、成交量、成交额
- 基本面：财报、市值、估值、盈利质量
- 文本信息：公告、新闻、研报、政策、社媒
- 关系结构：行业、概念、供应链、资金联动、事件传播
- 交易上下文：滑点、流动性、持仓、可交易状态、风控限制

在当前项目中，对应基础模块主要是：

- [utils/data_fetcher.py](D:/quant_trading_project/utils/data_fetcher.py)
- [utils/a_share_minute_data.py](D:/quant_trading_project/utils/a_share_minute_data.py)
- [factors/a_share_factors.py](D:/quant_trading_project/factors/a_share_factors.py)

### 4.2 特征与表示层

这一层不直接输出交易，而是把不同模态编码成可融合表示。

建议拆成五个编码器：

- `tabular encoder`
  - 服务日频因子、基本面、风格暴露
  - 当前强基线：`LightGBM / XGBoost`

- `time-series encoder`
  - 服务分钟级和日频序列
  - 候选：`iTransformer / PatchTST / ModernTCN / Autoformer`

- `text encoder / MLLM`
  - 服务公告、新闻、研报、问答
  - 候选：`FinGPT / BloombergGPT` 路线

- `graph encoder`
  - 服务行业图、概念图、事件传播图
  - 候选：`GAT / GraphSAGE / TGN`

- `state encoder`
  - 服务市场状态、风格状态、风险状态
  - 与 world model 共用 latent state

### 4.3 记忆层

`Memory` 是 M3-Net 的关键，不建议只理解成 LLM 上下文记忆。

这里至少应包含四类记忆：

- `market memory`
  - 保存市场 regime、波动状态、轮动节奏

- `alpha memory`
  - 保存近期有效因子、近期失效因子、信号稳定性

- `execution memory`
  - 保存滑点、成交效率、下单延迟、失败订单模式

- `research memory`
  - 保存论文笔记、实验结论、失败案例、策略解释模板

其中 `research/paper_notes` 已经是项目里的第一版研究记忆库。

### 4.4 专家路由层

MoE 层不应只按模型结构来拆，而要按金融任务拆专家。

建议初版专家划分：

- `daily_alpha_expert`
  - 日频横截面 alpha

- `intraday_alpha_expert`
  - 分钟级短线 alpha

- `event_text_expert`
  - 新闻/公告/研报事件语义

- `graph_relation_expert`
  - 图结构风险传播与关系 alpha

- `risk_control_expert`
  - 风险暴露、极端市场保护

- `execution_expert`
  - 下单、滑点、成交概率优化

路由输入建议包含：

- 市场状态
- 时间尺度
- 任务类型
- 标的类别
- 文本事件强度
- 波动 regime

早期路由可以先用简单 gating MLP，不必一开始就做全量复杂 MoE。

### 4.5 决策层

决策层负责把多专家输出转成统一的可交易结果。

建议拆成三种决策头：

- `score head`
  - 输出个股 alpha score
  - 用于选股、排序、组合构建

- `state head`
  - 输出市场状态、事件状态、风格状态
  - 用于仓位控制与风控

- `policy head`
  - 输出连续仓位、交易强度、执行动作
  - 候选：`PPO / SAC`

### 4.6 组合与执行层

这一层是项目盈利能力的真正落点。

包括：

- 股票池过滤
- 风格/行业约束
- 组合优化
- 等权/风险平价/目标暴露构建
- 交易日历与停牌过滤
- 涨跌停过滤
- 下单模拟
- 成交回报记录

当前项目中可扩展的现有基础：

- [factors/a_share_factors.py](D:/quant_trading_project/factors/a_share_factors.py)
- [strategies/market_neutral.py](D:/quant_trading_project/strategies/market_neutral.py)
- [strategies/intraday_t0.py](D:/quant_trading_project/strategies/intraday_t0.py)
- [backtest/engine.py](D:/quant_trading_project/backtest/engine.py)

### 4.7 评估与治理层

这一层必须独立存在，不能混在训练脚本里。

包括：

- 时间切分验证
- 滚动回测
- 样本外测试
- 模拟盘评估
- 归因分析
- 因子稳定性分析
- 专家负载分析
- 数据泄漏检测
- 风险告警

## 5. 当前项目中的落地映射

## 5.1 已有能力

当前项目已经具备以下基础：

- 日频与分钟级数据接入
- A 股多因子基础计算
- `LightGBM / XGBoost` 因子模型
- 单标的回测基础设施
- T0 策略雏形
- 论文资料与研究笔记体系

## 5.2 缺失能力

距离 M3-Net 还有这些关键空白：

- 文本数据接入与金融文本编码
- 多模态表示融合
- 专家路由系统
- 市场状态 world model
- 图结构关系建模
- 因果推断专用模块
- 神经符号约束模块
- 多资产/多股票组合级回测引擎
- 执行层 RL 模块

## 6. 推荐目录设计

建议未来按下面结构扩展：

```text
quant_trading_project/
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ cache/
│  └─ feature_store/
├─ factors/
├─ models/
│  ├─ tabular/
│  ├─ sequence/
│  ├─ multimodal/
│  ├─ moe/
│  ├─ world_model/
│  ├─ graph/
│  ├─ policy/
│  └─ causal/
├─ memory/
│  ├─ market_memory/
│  ├─ alpha_memory/
│  ├─ execution_memory/
│  └─ research_memory/
├─ strategies/
├─ portfolio/
├─ execution/
├─ backtest/
│  ├─ single_asset/
│  ├─ cross_sectional/
│  └─ execution_sim/
├─ research/
│  ├─ papers/
│  ├─ paper_notes/
│  └─ M3_NET_PROJECT_ARCHITECTURE.md
└─ tests/
```

## 7. 模型组合建议

## 7.1 第一阶段：强基线

先不要直接上完整 M3-Net。

第一阶段推荐：

- 日频主模型：`LightGBM`
- 对照模型：`XGBoost`
- 分钟级主模型：`iTransformer / PatchTST / ModernTCN`
- 组合层：等权 + 基础风控

目的：

- 确立基线
- 确立评估协议
- 确立交易成本模型

## 7.2 第二阶段：轻量 M3-Net

引入：

- 文本编码器
- 一个简单 MoE 路由器
- 一个简化市场状态 world model

不建议这阶段做：

- 全量 RL
- 全量因果推断
- 全量神经符号推理

## 7.3 第三阶段：完整 M3-Net

在基线稳定之后，再加入：

- 图编码器
- 执行层 RL
- 因果鲁棒验证
- 神经符号风控

## 8. 各核心组件的职责边界

### 8.1 MLLM

负责：

- 公告/新闻/研报理解
- 金融问答
- 事件摘要
- 因子解释
- 研究辅助

不建议直接负责：

- 单独输出最终交易指令
- 替代全部数值模型

### 8.2 MoE

负责：

- 按任务和市场状态分工
- 让不同专家处理不同模态和不同时间尺度

不建议直接负责：

- 替代所有 backbone
- 在没有强基线前单独训练

### 8.3 World Model

负责：

- 市场 latent state
- regime 切换
- 情景模拟
- imagined rollout

不建议直接负责：

- 直接替代所有预测器

### 8.4 Causal ML

负责：

- 检查信号是否只是伪相关
- 做干预、反事实、稳健性验证

当前注意：

- 你的资料包里还缺真正专门的 Causal ML 论文
- 当前更适合把它当作后续增强层，而不是第一阶段主模型

### 8.5 Neuro-Symbolic

负责：

- 把模型输出与显式规则结合
- 强化风控、合规、仓位边界、停牌和涨跌停约束

当前注意：

- 现有论文包主要提供图神经网络基础
- 还需要后续补专门的规则推理/符号约束论文

## 9. 数据流设计

建议统一数据流：

1. 原始数据进入 `raw`
2. 清洗、对齐、补齐后进入 `processed`
3. 特征、事件、图结构进入 `feature_store`
4. 各编码器生成 embedding / score / state
5. MoE 路由组合各专家结果
6. 决策层输出 score / action / position
7. 组合层构建持仓
8. 执行层模拟成交
9. 回测层评估收益、风险、稳定性
10. 结果写回 memory 层

## 10. 训练流设计

训练建议分四条线并行推进：

### 10.1 日频训练线

- 数据：日频行情 + 基本面 + 因子
- 标签：未来 `N` 日收益 / 超额收益
- 基线：`LightGBM / XGBoost`

### 10.2 分钟级训练线

- 数据：分钟行情 + 微结构特征
- 标签：短期收益、回撤、成交质量
- backbone：`iTransformer / PatchTST / ModernTCN`

### 10.3 文本训练线

- 数据：公告、新闻、研报、舆情
- 输出：事件标签、情绪分数、摘要 embedding
- backbone：金融 MLLM

### 10.4 状态/策略训练线

- 数据：多模态融合后的 latent state
- 输出：regime、风险状态、仓位策略
- backbone：world model + RL

## 11. 评估体系

每条模型线都要分别评估。

建议最少包含：

- 收益率
- 年化收益
- Sharpe
- Sortino
- 最大回撤
- Calmar
- 换手率
- 胜率
- IC / RankIC
- 风格暴露
- 滑点敏感性
- 样本外稳定性
- 回测与模拟盘一致性

MoE 还要额外看：

- 专家负载是否均衡
- 专家是否塌缩
- 不同 regime 下的激活分布

World model 还要额外看：

- latent state 是否稳定
- regime 预测是否可解释
- imagined rollout 与真实市场是否偏离过大

## 12. 风险与边界

### 12.1 主要风险

- 模型过于复杂，超过当前数据质量和工程能力
- 多模态融合导致噪声放大
- 文本信号带来伪相关
- world model 产生过度自信的虚拟市场
- RL 在不准确执行模拟器上学到错误策略

### 12.2 控制原则

- 所有复杂模块必须有简单控制组
- 所有模块必须能被单独关掉
- 所有模块必须做样本外和滚动验证
- 所有交易结果必须经过显式风控层

## 13. 推荐实施顺序

### 阶段 A

- 强化现有 `LightGBM / XGBoost` 因子模型
- 完善组合回测
- 完善分钟级特征层

### 阶段 B

- 引入时序 backbone
- 引入金融文本编码
- 建立统一 feature store

### 阶段 C

- 引入轻量 MoE
- 引入 market state encoder
- 建立第一版 world model

### 阶段 D

- 引入执行层 RL
- 引入图结构动态关系建模
- 引入因果与符号约束

## 14. 本项目的最终架构结论

对这个项目来说，`M3-Net` 最好的定位不是“一次性大重构”，而是一个分层升级目标。

短期内：

- 用 `LightGBM / XGBoost` 站稳脚跟
- 用分钟级时序模型补足短线能力

中期：

- 用文本、图和市场状态把信号源扩展成多模态
- 用轻量 MoE 做分工和融合

长期：

- 用 world model、RL、causal、neuro-symbolic 形成真正具备研究、预测、执行和自我修正能力的 M3-Net 系统

这会比直接“上一个大模型”更稳，也更符合你当前项目的工程现实。
