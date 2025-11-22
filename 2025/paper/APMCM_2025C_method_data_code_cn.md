# APMCM 2025 C 题建模方法、代码与数据说明

## 摘要

围绕 2025 年美国实施的“互惠关税”（Reciprocal Tariffs）政策，本文建立了涵盖大豆贸易（Q1）、日本汽车（Q2）、半导体供应链（Q3）、关税收入拉弗曲线（Q4）以及宏观金融与制造业回流（Q5）在内的多层次建模体系。在数据层面，我们从 UN Comtrade、FRED、BEA、BLS、IMF 等官方或权威来源构建统一的数据仓库，并通过清洗、对齐和指数化处理形成可直接用于建模的年度与月度面板。在方法层面，我们结合经济计量模型（OLS、面板回归、VAR）、结构模型、多智能体博弈（MARL）、图神经网络（GNN）以及 VAR-LSTM 混合模型，对关税冲击在贸易流、产业结构、财政收入以及宏观金融变量上的传导机制进行刻画与情景模拟。

在代码实现层面，项目采用模块化结构，将各题模型分别封装于 `2025/src/models` 目录下的独立模块，通过统一的数据加载与结果导出接口，实现从原始数据获取到可视化与报告生成的自动化流程。本文档重点说明数据结构设计、核心建模思想以及各模块代码之间的调用关系，为正式论文“数据与方法”及技术附录提供统一参考。

**关键词：** 互惠关税；大豆贸易；日本汽车；半导体供应链；关税收入；VAR；LSTM；GNN；多智能体博弈

## 1 引言

近年来，以美国为代表的大国相继推出关税调整、产业补贴与出口管制等一揽子政策，全球贸易和供应链格局面临显著重塑。2025 年提出的“互惠关税”制度进一步强化了对对手国的差别化关税安排，并与制造业回流、关键技术安全等政策目标相结合。赛题 C 题在这一背景下提出了五个具体问题，要求围绕大豆、日本汽车、半导体、关税收入以及宏观金融与制造业回流构建一体化分析框架。

本项目在严格遵守赛题数据与时间约束的前提下，以“机制为主、预测为辅”为总体建模思路。一方面，通过结构化的经济理论模型（弹性估计、拉弗曲线、VAR 模型等）确保模型结论具有清晰的经济含义和可解释性；另一方面，引入 GNN、Transformer、VAR-LSTM 等机器学习方法，对非线性关系与高维交互进行补充刻画，同时利用正则化、交叉验证和小样本稳健性约束抑制过拟合风险。

在工程实现层面，我们将五个子问题统一纳入同一代码与数据框架，通过标准化的数据路径（`DATA_EXTERNAL`、`DATA_PROCESSED`）与结果路径（`RESULTS_DIR`、`FIGURES_DIR`），实现“同一数据源，多模型复用”的工程目标。本文档对数据体系、建模方法以及代码组织结构进行整体梳理，并给出模型与脚本的对应关系，为后续论文撰写和结果复现实验提供技术依据。

## 2 数据与预处理

### 2.1 数据总体框架

本项目的数据分为三类：

- **外部原始数据（external）**：来源于官方或权威公开数据库，如 UN Comtrade、FRED、BEA、BLS、IMF 等，存放于 `2025/data/external` 目录。
- **中间处理数据（interim）**：在预处理和特征构造过程中产生的中间结果，存放于 `2025/data/interim` 目录。
- **建模用处理后数据（processed）**：经过清洗、对齐、单位统一和指标构造后的最终建模数据，存放于 `2025/data/processed` 目录，并按 Q1–Q5 分子文件夹管理。

数据预处理遵循以下统一原则：

- **时间维度统一**：根据各子问题需求，分别采用月度（Q1）和年度（Q1–Q5）频率，确保时间轴连续且无缺口。
- **单位与币种统一**：金额统一为美元（USD），数量统一为吨或指数形式；对不同来源数据进行汇率和物价调整。
- **指标标准化与可比性**：构造指数（如 `tariff_index_total`、`global_chip_demand_index` 等）时，以基期归一化为 100 或 1，便于跨期比较。
- **缺失值处理**：对关键变量采用插值、滚动均值或简单情景假设处理，并在文中说明假设与局限性。

### 2.2 Q1 大豆贸易数据

Q1 主要使用两类数据：

- **中国自美国、巴西、阿根廷进口大豆月度数据**：来源于 UN Comtrade，经整理后形成 `2025/data/processed/q1/q1_1.csv`。
  - 主要字段包括 `period`（YYYYMM）、`partner_desc`（USA/Brazil/Argentina）、`net_weight_tons`、`primary_value_usd`、`tariff_rate`、`data_quality_score` 等。
  - 时间范围约为 2010–2024 年，覆盖三大出口国月度双边贸易。
- **年度汇总面板**：`2025/data/processed/q1/q1_0.csv`，包含 2015–2024 年三国对华大豆出口数量、金额、单价以及中国对各国关税水平，用于弹性估计和结构性比较。

预处理在脚本 `2025/src/utils/external_data.py` 和相关数据准备脚本中完成，主要步骤包括：

- 按 `partner_desc` 统一国家命名，剔除异常小额记录或测试记录；
- 将净重从千克换算为吨，并构造 `unit_value`、`price_with_tariff` 等派生变量；
- 将关税水平映射为 `tariff_rate`，并根据政策情景（互惠关税）构造未来期间的关税路径。

### 2.3 Q2 日本汽车数据

Q2 使用的主要数据文件位于 `2025/data/processed/q2` 以及 `2025/data/external/q2`：

- 品牌层面销售与进口结构数据 `q2_1.csv`（processed），包含 2015–2024 年日本品牌在美国市场的销量、价格、市场份额等。
- 行业与宏观指标来自 `external` 文件夹，如美国总汽车销量指数、工业生产指数等。

数据预处理由 `2025/src/utils/external_data.py` 和专门的数据准备脚本完成，主要步骤包括：

- 统一品牌名称与车型分类，将原始网站数据（如 goodcarbadcar）转化为年度品牌面板；
- 计算市场份额、价格指数等派生变量，为后续结构模型与多智能体博弈提供输入；
- 对宏观控制变量进行标准化，以避免量纲差异对估计产生影响。

### 2.4 Q3 半导体供应链数据

Q3 的处理后数据主要包括：

- `2025/data/processed/q3/q3_1.csv`：半导体行业年度主面板。
  - 含 `year`、`us_chip_output_index`、`us_chip_output_billions`、`global_chip_output_billions`、`us_global_share_pct`、`global_chip_demand_index`、`china_import_dependence_pct`、`policy_support_index`、`export_control_index`、`supply_chain_risk_index` 等变量。
  - 覆盖 2015–2024 年 10 个年度观测。
- `2025/data/processed/q3/q3_2_supply_chain_segments.csv`：分 segment（高端、中端、低端）供应链面板，用于构建图结构和 GNN 特征。

预处理与特征构造在 `2025/scripts/generate_processed_q345.py` 中实现，主要包括：

- 汇总全球与美国芯片产出数据，计算美国全球份额及中国进口依赖度；
- 将政策信息（补贴、出口管制、回流激励等）指数化，形成年度政策强度指标；
- 构建供应链图节点与边特征，为 GNN 模型输入做准备。

### 2.5 Q4 关税收入与拉弗曲线数据

Q4 使用的数据包括：

- `2025/data/processed/q4/q4_0_tariff_revenue_panel.csv`：年度关税收入面板，包含 `year`、`total_imports_usd`、`total_tariff_revenue_usd`、`effective_tariff_rate` 等，用于估计静态与动态拉弗曲线；
- `2025/data/processed/q4/q4_1_tariff_scenarios.csv`：不同政策情景下的平均关税率与进口规模路径，用于收入模拟和情景比较。

部分原始信息来自 `2025/data/external/q4_avg_tariff_by_year.csv`，通过 `TariffDataLoader` 类（`2025/src/utils/data_loader.py`）进行清洗与汇总。

### 2.6 Q5 宏观金融与制造业回流数据

Q5 使用的数据集中存放于：

- `2025/data/processed/q5/q5_4_integrated_panel.csv`：集成宏观、金融、关税指数和回流事件的年度面板，包含 `year`、`gdp_growth`、`unemployment_rate`、`industrial_production`、`tariff_index_total`、`retaliation_index`、`sp500_index`、`manufacturing_va_share` 等字段；
- 关税强度指数文件（如 `q5_tariff_indices_calibrated.csv`、`q5_tariff_indices_policy.csv` 等），由 `2025/scripts/make_q5_tariff_indices.py` 生成。

外部宏观数据则来自 `2025/data/external` 下的 FRED 与其他官方来源（如 `q5_us_macro_from_grok4_candidate.csv` 等），通过统一脚本进行合并与指标构造。

## 3 建模方法概述

本项目的整体方法论可以概括为“理论结构 + 小样本计量 + 机器学习增强”。对于每一个子问题，我们都首先给出一个具有清晰经济含义的“主方程”或结构框架，在此基础上利用有限样本进行参数估计或标定；随后再根据需要引入 GNN、Transformer、VAR-LSTM 等机器学习模块，提高对复杂非线性关系和情景组合的刻画能力。本节从 Q1–Q5 依次简要说明各题目的核心建模思路。

### 3.1 Q1：大豆贸易弹性与情景模拟

Q1 的目标是刻画中国从美国、巴西和阿根廷三国进口大豆时，关税与相对价格变化如何在中短期内影响贸易量与市场份额。我们在年度面板和月度面板上分别建立两个层次的模型：

- **年度弹性模型（结构层）**：在 `q1_0.csv` 上构造年度面板，以年为时间维度、出口国为截面单位。核心方程形如：
  
  \[\ln Q_{i,t} = \alpha_i + \gamma_t + \beta_1 \ln P^{\text{tariff}}_{i,t} + \beta_2 T_{i,t} + u_{i,t},\]
  
  其中 \(Q_{i,t}\) 为从出口国 \(i\) 进口的大豆数量，\(P^{\text{tariff}}_{i,t}\) 为含税单价，\(T_{i,t}\) 为对该国的名义关税率，\(\alpha_i\) 为出口国固定效应，\(\gamma_t\) 为时间固定效应。`SoybeanTradeModel.estimate_trade_elasticities` 在内部自动构造对数变量与固定效应，并给出价格弹性与相对关税弹性的估计值及显著性检验。

- **相对份额模型**：在计算各国市场份额后，构造相对美国的份额对数比 \(\ln(s_i/s_{US})\) 作为因变量，以关税差异 \(T_i - T_{US}\) 作为核心解释变量，用于衡量相对关税变化对份额替代的影响。该部分仍在 `estimate_trade_elasticities` 中自动完成。

- **月度情景模拟（动态层）**：在 `q1_1.csv` 月度数据基础上，`SoybeanTradeModel.simulate_tariff_scenarios` 选取最近若干年的观测作为基线，基于估计得到的价格弹性，对“基准情景、互惠关税情景、极端报复情景”等不同关税路径进行前推，计算各出口国的进口量和市场份额变化。`plot_q1_results` 进一步输出各情景下的份额柱状图，展示美国与巴西、阿根廷之间的替代关系。

同时，为了在时间维度上更细致地刻画月度动态，我们在 `SoybeanMonthlyDataset` 和 `SoybeanDataProcessor` 中实现了 LSTM 所需的序列化特征工程，包括季节特征（`month_sin`/`month_cos`）、多阶滞后与滚动均值、关税与价格弹性 proxy 等，为后续深度学习预测模块提供输入。

### 3.2 Q2：日本汽车多智能体博弈与机器学习预测

Q2 关注日本汽车企业在美国提高对日关税后，如何在“继续出口—在美生产—在墨西哥生产”等多种生产布局之间进行权衡。我们采用“结构计量 + 情景模拟 + 博弈分析 + Transformer 预测”的四步框架：

- **进口结构计量模型**：`AutoTradeModel.estimate_import_structure_model` 在关税数据不完全的前提下，构造以关税代理变量（`auto_import_charges`）为基础的份额模型 \(\ln(s_j) = \delta_j + \phi_1 t_j + \phi_2 X_t + e_{j,t}\)，以时间趋势和合作伙伴虚拟变量近似刻画关税与宏观因素对进口份额的长期影响。

- **产业传导模型**：`estimate_industry_transmission_model` 将“进口渗透率（import_penetration）”与美国汽车产量、就业等行业指标相联系，估计 \(Y^{US}_t = \theta_0 + \theta_1 \text{ImportPen}_t + \theta_2 Z_t + \nu_t\)，从而将日本汽车进口情景映射到美国本土生产和就业变化上。

- **情景模拟与结果导出**：`simulate_japan_response_scenarios` 设定“无响应、部分迁移、激进本地化”三种代表性情景，分别给出美国生产、墨西哥生产、日本直接出口的份额，计算对应的进口渗透率和产量/就业变化，并将结果导出为 CSV 与 JSON，供博弈分析和论文制图使用。

- **多智能体博弈与 Nash 均衡分析**：`NashEquilibriumSolver` 与 `AutoTradeModel.run_marl_analysis` 在上述情景结果基础上，构造美国关税率与日本迁移强度的策略格子，利用收益函数（美国：就业 + 财政收入 − 消费者成本；日本：利润保持 − 迁移成本 + 市场份额收益）搜索纯策略 Nash 均衡，并给出帕累托最优均衡集及政策推荐。

- **Transformer 时序预测模块**：`prepare_transformer_sequence_data` 和 `build_transformer_model` 将年度进口负担 `auto_import_charges` 转化为滑动窗口序列，训练轻量级 Transformer 结构用于预测未来进口负担路径，与 OLS/情景分析形成互补。

### 3.3 Q3：半导体供应链 GNN 风险评估

Q3 的核心是量化半导体供应链在不同政策组合下的“效率–安全”权衡。我们在横截面和网络结构两个层面上进行建模：

- **分 segment 的贸易与产出计量模型**：`SemiconductorModel.load_q3_data` 与 `estimate_trade_response` 在高端、中端、低端三个 segment 上分别估计 \(\ln M_{s,j,t} = \alpha_{s,j} + \beta_s t + e_{s,j,t}\) 形式的趋势模型，用于刻画各 segment 进口强度的时间演变；`estimate_output_response` 则在 `q3_1.csv` 输出与政策数据合并后，估计补贴指数、出口管制指数等变量对美国半导体产出的弹性。

- **供应链安全指标构造**：`compute_security_metrics` 在 `q3_2_supply_chain_segments.csv` 基础上，计算各 segment 的“自给率（self_sufficiency_pct）”“中国依赖度（china_dependence_pct）”及综合的 `supply_risk_index`，为后续政策情景比较提供基线。

- **GNN/图分析模块**：在 GNN 可用时，`run_gnn_analysis` 调用 `q3_gnn_tri` 或 `q3_gnn` 中的完整异质图 GNN，学习供应链图中的隐含结构与风险传播；在算力或依赖受限时，退化为 `SupplyChainGraph` 的简化图分析，直接计算集中度（HHI）、地缘政治风险加权平均、技术集中度和安全指数，并通过 `simulate_disruption` 模拟对关键供应国（如中国、台湾、韩国）发生不同强度中断时的级联影响。

- **政策组合效率–安全前沿**：`simulate_policy_combinations` 使用来自产出弹性与安全指标的结果，为“补贴优先、关税优先、综合政策”等情景计算自给率提升、安全指数变化与成本指数，绘制效率–安全前沿曲线，为政策评价提供定量依据。

### 3.4 Q4：关税收入拉弗曲线与政策情景

Q4 旨在通过有限年度样本，构造一个既具有经济含义又可用于情景推演的关税收入拉弗曲线框架：

- **静态拉弗曲线估计**：`TariffRevenueModel.estimate_static_revenue_model` 基于 `q4_0_tariff_revenue_panel.csv` 和年度平均关税率 `avg_tariff`，估计 \(\ln R_t = \alpha + \beta_1 \tau_t + \beta_2 \tau_t^2 + \varepsilon_t\)，并据此计算最优关税率 \(\tau^* = -\beta_1/(2\beta_2)\) 及其可信区间，以说明“过高关税可能反而降低收入”的拉弗曲线机制。

- **动态进口响应估计**：`estimate_dynamic_import_response` 通过对进口额对数差分与关税变动的一阶/二阶差分回归，估计短期和中期弹性，用于在情景模拟中调整进口基数（而非简单假定进口不变）。在样本不足时，该模块也支持从外部 JSON 读取手工设定的动态参数。

- **第二任期收入情景模拟**：`simulate_second_term_revenue` 结合静态与动态弹性，从 `q4_1_tariff_scenarios.csv` 或 `q4_tariff_scenarios.json` 读取一组关税路径，包括基准、温和互惠关税、激进贸易战、去冲突等 5–7 种情景，给出 2025–2029 年的收入路径与累积收入差异，并在结果中对比基准情景与互惠关税情景的净财政收益。

- **机器学习增强与模型比较**：`train_ml_models` 和 `train_arima_model` 在同一年度序列上训练梯度提升树与 ARIMA 模型，用于拟合收益对关税路径的非线性响应；`ml_forecast_revenue` 与 `compare_models` 对比计量模型和 ML 模型的总收入预测与误差结构，验证拉弗曲线结论在不同方法下的稳健性。

### 3.5 Q5：VAR-LSTM 宏观金融模型与制造业回流

Q5 的重点在于刻画“关税与报复—宏观与金融—制造业回流”之间的联动机制。鉴于年度样本较少，我们采用“小维度 VAR + 回归 + VAR-LSTM + ML 解释”的分层方案：

- **回归层：单方程效应估计**。`estimate_regression_effects` 针对 \(Y_t \in\{\text{gdp_growth}, \text{industrial_production}, \text{manufacturing_va_share}, \dots\}\) 构造 \(Y_t = \lambda_0 + \lambda_1 \text{TariffIndex}_t + \lambda_2 \text{RetaliationIndex}_t + \lambda_3 Z_t + \varepsilon_t\) 的线性回归，用于给出关税指数和报复指数的方向性影响与显著性。

- **VAR 层：多变量动态系统**。`estimate_var_model` 在关税指数、报复指数、GDP 增速、工业生产等变量上估计 VAR 模型，并计算关税冲击的脉冲响应函数（IRF），用于展示“关税上升—增长下滑—生产调整”这一动态链条的时间路径。

- **事件研究 / DID：制造业回流评估**。`evaluate_reshoring` 通过在 2025 年设置“互惠关税实施”的处理时间点，对制造业增加值占比进行前后比较和带趋势控制的回归，给出回流效应的数量级和统计显著性。

- **VAR-LSTM 混合模型**。`train_var_lstm_hybrid` 首先以 VAR 模型拟合线性联动，得到残差序列；随后将残差序列输入 LSTM 网络，通过序列预测学习剩余非线性部分。在网络结构中显式加入 dropout、L2 正则和 early stopping，并在小样本条件下控制窗口长度和批大小，以降低过拟合风险。最终得到的 MSE/RMSE 作为“非线性部分能否被有效学习”的指标。

- **机器学习解释层：制造业回流 ML 模型**。`train_reshoring_ml` 以制造业增加值占比为因变量、以关税指数、报复指数、宏观金融变量及其滞后为特征，训练随机森林和梯度提升树模型，并输出特征重要性排序，用于解释“哪些变量最能解释制造业回流”的数据驱动结论。

## 4 代码结构与运行方式

### 4.1 目录结构概述

项目的代码与数据组织遵循模块化和可复现原则，核心结构如下（部分）：

- `2025/data/external`：外部原始数据文件；
- `2025/data/processed`：按 Q1–Q5 划分的处理后数据；
- `2025/src/models`：各题目的模型实现文件，如 `q1_soybeans.py`、`q2_autos.py`、`q3_semiconductors.py`、`q4_tariff_revenue.py`、`q5_macro_finance.py` 等；
- `2025/src/utils`：数据加载与工具函数，包括 `data_loader.py`、`external_data.py` 等；
- `2025/scripts`：数据预处理和索引构造脚本，如 `generate_processed_q345.py`、`make_q5_tariff_indices.py`；
- 统一运行脚本（如 `run_all_models.py` 或 `main.py`）用于一键运行全部或部分模型。

### 4.2 依赖与环境管理

项目使用 Python 作为主要语言，依赖通过 `pyproject.toml` 与 `uv.lock` 进行管理。主要第三方库包括：

- 数据处理：`pandas`、`numpy`；
- 计量与统计：`statsmodels`、`scikit-learn`；
- 时间序列与深度学习：`torch`（或 `tensorflow`）、可能配合 `pytorch-forecasting` 等；
- 可视化：`matplotlib`、`seaborn`、网络图相关库等。

环境准备通常通过：

```bash
uv sync
```

完成依赖安装，然后可使用 `uv run` 运行各个脚本。

### 4.3 模型运行示例

在命令行或 PowerShell 中，可使用类似命令运行模型（以实际脚本为准）：

- 运行数据准备脚本：

```bash
uv run python 2025/src/preprocessing/prepare_data.py
```

- 只运行 Q1 模型：

```bash
uv run python 2025/src/main.py --questions Q1
```

- 运行全部模型但关闭机器学习增强：

```bash
uv run python 2025/src/main.py --no-ml
```

或使用 `run_all_models.py` 统一生成结果与可视化。

## 5 可复现性与局限性

### 5.1 可复现性

为提高结果的可复现性，我们在代码中统一设置随机种子（如 42），并在：

- 数据预处理脚本中固定抽样与随机扰动；
- 机器学习与深度学习模型中固定权重初始化与划分方式；
- 结果导出时记录关键参数与时间戳。

此外，项目提供了若干检查与报告文档（如 `Model_Deep_Check_Report.md` 等），记录模型运行状态和已知问题。

### 5.2 局限性

需要坦诚的局限性主要包括：

- 部分问题（尤其 Q3–Q5）的年度样本数量有限，模型更偏向“结构解释 + 情景模拟”，统计显著性和预测精度存在一定限制；
- 多智能体博弈和 GNN 模型的参数空间较大，在参赛时间与算力约束下，无法进行全面的超参数寻优；
- 某些宏观或行业指标采用了指数或代理变量（proxy），可能与真实经济过程存在偏差。

在正式论文中，我们将结合本说明，从数据、模型和实现三个层面综合评估结果的可靠性和适用范围。

## 6 小结

本文档系统介绍了 APMCM 2025 C 题项目的数据体系、建模方法与代码实现结构。整体上，我们构建了从原始数据获取、预处理与指标构造，到 Q1–Q5 各子问题模型估计与情景模拟的完整工作流，并通过统一的数据/结果目录约定和模块化代码设计，提高了工程实现的可维护性与可复现性。

在此基础上，正式论文的“数据与方法”部分将以本说明为蓝本，结合各题目的详细技术附录，对关键变量定义、估计方法和情景设置进行更为深入的论述；评审或读者若需进一步了解实现细节，可直接参考本说明中的代码路径与附录结构，快速定位到对应模型与脚本。

## 7 附录：代码与方法对应关系

本附录从“代码—数据—方法”三条线索出发，简要说明各题目模型与主要类/函数的对应关系，方便查阅与复现。

### 7.1 Q1 大豆贸易模型（`q1_soybeans.py`）

- **`SoybeanTradeModel`**：
  - 负责从关税数据中提取与大豆相关的出口记录（`load_q1_data`），并通过 `HSMapper` 标记大豆产品；
  - `load_external_china_imports` 以 `q1_1.csv` 为统一“单一数据源”，标准化出口国名称、构造年度进口面板，并生成 `unit_value`、`price_with_tariff` 等关键变量；
  - `prepare_panel_for_estimation` 进一步构造对数变量、市场份额及相对关税差异，形成用于弹性估计的面板数据框。
- **`estimate_trade_elasticities`**：在上述面板基础上自动估计“数量—价格”模型与“份额—关税差异”模型，返回包含弹性系数、标准误与显著性检验的字典，并将完整回归结果以 JSON 形式保存到 `results/q1/q1_elasticities.json`。
- **`simulate_tariff_scenarios` 与 `plot_q1_results`**：
  - 前者基于最近年度的基线数据，使用估计得到的弹性对“基准、互惠关税、完全报复”等情景进行数量与份额模拟，并将结果写入 `q1_scenario_exports.csv`；
  - 后者读取该 CSV，生成不同情景下美、巴、西三国市场份额对比图，输出为 `figures/q1_shares_before_after.pdf`。
- **`SoybeanMonthlyDataset.load` 与 `SoybeanDataProcessor`**：
  - `SoybeanMonthlyDataset.load` 对 `q1_1.csv` 做月度层面的日期解析、出口国标准化、数量与金额单位统一，并对关税列进行向前填充，形成完整月度时间序列；
  - `SoybeanDataProcessor.prepare_features` 和 `build_supervised_arrays` 在此基础上构造季节特征、滞后与滚动均值、关税冲击和价格弹性 proxy，并完成归一化与监督学习样本生成，为 LSTM 等深度学习模型提供输入。

### 7.2 Q2 日本汽车模型（`q2_autos.py`）

- **`AutoTradeModel`**：
  - `load_q2_data` 使用 `TariffDataLoader` 和 `HSMapper` 从关税数据中提取与汽车相关的进口记录，并按年份与贸易伙伴汇总形成 `auto_import_charges` 面板；
  - `load_external_auto_data` 从 `data/external` 中读取或自动生成品牌销量和行业指标模板，为结构模型与产业传导模型提供外生变量。
- **经济计量部分**：
  - `estimate_import_structure_model` 构造进口份额对数模型，估计时间趋势与国别固定效应，用于描述日本和其他国家在美国汽车市场的结构演变；
  - `estimate_industry_transmission_model` 结合“进口渗透率”和行业指标（产量、价格指数、GDP 等），估计进口对美国本土生产和就业的传导效应。
- **情景分析与博弈部分**：
  - `simulate_japan_response_scenarios` 以参数化形式设定三种典型情景（无响应、部分迁移、激进本地化），输出对应的进口渗透率和产量/就业变化；
  - `NashEquilibriumSolver` 与 `run_marl_analysis` 读取情景结果，构造美国关税率与日本迁移强度的策略空间，计算博弈双方的收益矩阵，搜索纯策略 Nash 均衡并输出 Markdown 报告。
- **机器学习增强部分**：
  - `prepare_transformer_sequence_data` 将 `auto_import_charges` 转化为按伙伴国分组的时间序列窗口；
  - `build_transformer_model` 定义轻量级 Transformer 编码器结构，用于拟合和预测未来的进口负担轨迹。

### 7.3 Q3 半导体模型（`q3_semiconductors.py`）

- **`SemiconductorModel`**：
  - `load_q3_data` 通过 HS 编码标记半导体产品，并按年份、贸易伙伴与 segment 聚合得到 `chip_import_charges`；
  - `load_external_chip_data` 读取或生成美国半导体产出与政策指数（补贴、出口管制等）数据，保证与 `q3_1.csv`/外部文件的一致性；
  - `estimate_trade_response` 与 `estimate_output_response` 分别在贸易和产出两端估计时间趋势与政策弹性。
- **安全指标与政策模拟**：
  - `compute_security_metrics` 在贸易与产出数据的基础上计算自给率、中国依赖度与供应风险指数；
  - `simulate_policy_combinations` 在给定补贴与关税组合假设下，模拟效率（产出）与安全（自给率/风险指数）指标的变化，并输出多情景结果表。
- **GNN/图分析模块**：
  - `SupplyChainGraph` 提供简化图表示，用于在 GNN 不可用时依然能够计算集中度、安全指数和中断影响；
  - `run_gnn_analysis` 在依赖满足时优先调用 `q3_gnn_tri` 或 `q3_gnn`，执行完整的异质图 GNN 训练，并生成风险分析 JSON 与 Markdown 报告。

### 7.4 Q4 关税收入模型（`q4_tariff_revenue.py`）

- **`TariffRevenueModel`**：
  - `load_q4_data` 优先读取处理好的年度关税收入面板 `q4_0_tariff_revenue_panel.csv`，在失败时回退到原始进口数据的年度汇总；
  - `estimate_static_revenue_model` 完成静态拉弗曲线估计，计算最优平均关税率并将结果导出为 JSON；
  - `estimate_dynamic_import_response` 优先使用处理后面板估计进口对关税变动的动态弹性，并在必要时从外部配置读取参数；
  - `simulate_second_term_revenue` 使用静态与动态弹性参数，将外部或处理后情景文件中的关税路径映射为 2025–2029 年的收入轨迹，并对比基准与政策情景的累计收入差异；
  - `train_ml_models`、`train_arima_model`、`ml_forecast_revenue` 与 `compare_models` 共同构成 ML 增强模块，对比计量与 ML 模型的预测表现和对关税路径敏感性的差异。

### 7.5 Q5 宏观金融与制造业回流模型（`q5_macro_finance.py`）

- **`MacroFinanceModel`**：
  - `load_q5_data` 从关税指数、宏观数据、金融数据、回流数据和报复指数五个来源中合并年度面板，形成包含关税、宏观、金融与制造业指标的统一时间序列；
  - `estimate_regression_effects` 对 GDP 增速、工业生产、制造业增加值占比等变量分别估计关税与报复指数的边际效应；
  - `estimate_var_model` 基于选定变量集估计 VAR 模型并输出关税冲击的脉冲响应；
  - `evaluate_reshoring` 以 2025 年为处理时间点，评估制造业回流的前后差异与显著性；
  - `train_var_lstm_hybrid` 在 VAR 残差上训练 LSTM 网络，捕捉非线性动态；
  - `train_reshoring_ml` 针对制造业回流指标训练随机森林和梯度提升模型，并输出特征重要性；
  - `generate_model_comparison` 汇总 VAR、VAR-LSTM 与 ML 模型的关键评价指标，形成宏观层面的“解释性 vs 预测性”比较报告。

通过上述附录，读者可以从“问题编号（Q1–Q5）→ 模型文件 → 关键类/函数 → 数据输入与输出”的链条快速定位项目中任一分析环节的具体实现，从而在有限篇幅的正式论文之外，完整把握本队在建模与工程实现方面的工作量与技术细节。
