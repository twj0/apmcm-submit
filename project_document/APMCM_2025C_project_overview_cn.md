# APMCM 2025 C 题项目总览与使用说明

## 1 文档用途

本文件面向阅卷老师、助教和复现实验使用者，汇总说明：

- **赛题背景与五个子问题（Q1–Q5）的核心目标**；
- **数据与代码目录结构**，以及每个问题对应的数据、模型与结果文件；
- **当前一次完整运行得到的主要定量结论（基于仓库中的样例/占位数据）**；
- **如何在本机复现结果及需要注意的局限性**。

详细的建模推导与文稿请参见：

- `2025/paper/APMCM_2025C_paper_cn.md`（论文主体草稿）；
- `2025/paper/APMCM_2025C_method_data_code_cn.md`（方法、数据与代码结构详细说明）。

---

## 2 项目结构与主要文件

项目根目录下与本题相关的主要子目录：

- `2025/problems/2025 APMCM Problem C.md`：赛题原文（英文）。
- `2025/data/processed/`：清洗后可直接建模的数据，按 `q1`–`q6` 分文件夹。
- `2025/src/models/`：五个子问题的主要模型代码：
  - `q1_soybeans.py`
  - `q2_autos.py`
  - `q3_semiconductors.py`
  - `q4_tariff_revenue.py`
  - `q5_macro_finance.py`
  以及用于批量运行的 PowerShell 脚本 `run_models.ps1`。
- `2025/results/`：模型运行结果，按 `q1`–`q5` 与方法（`econometric/`、`marl/`、`gnn/`、`ml/` 等）细分；每个子目录包含 JSON/CSV/Markdown 报告。
- `2025/paper/`：中文论文及“数据与方法说明”草稿。
- `project_document/`：本文件等项目级汇总说明。

---

## 3 五个子问题概览

### 3.1 Q1：大豆贸易再分配与关税弹性

- **赛题目标**  
  在中国对美国大豆加征关税的背景下，分析美国、巴西和阿根廷三国对华大豆出口的再分配效应，估计贸易量和市场份额对含税价格和关税差异的弹性。

- **主要数据文件**  
  - `2025/data/processed/q1/q1_1.csv`：月度大豆进口面板，包含 `period`、`exporter`、`import_quantity`、`primary_value_usd`、`tariff_rate` 等字段。

- **核心代码与函数**  
  文件：`2025/src/models/q1_soybeans.py`
  - `SoybeanTradeModel.estimate_trade_elasticities()`：
    - 构造数量–价格面板回归和份额–关税差异回归；
    - 输出到 `2025/results/q1/q1_elasticities.json`。
  - `SoybeanTradeModel.simulate_tariff_scenarios()`：
    - 基于估计得到的弹性，对 `baseline`、`reciprocal_tariff`、`full_retaliation` 等情景进行数量与份额模拟；
    - 结果保存为 `2025/results/q1/q1_scenario_exports.csv`。

- **关键结果（基于当前样例数据）**  
  见 `results/q1/q1_elasticities.json` 与 `q1_scenario_exports.csv`：
  - 价格弹性约为 `price_elasticity ≈ -0.52`，显著为负（p 值约 0.005），说明中国自某一出口国的大豆进口额对含税价格具有中等程度的负向响应。  
  - 相对份额弹性约为 `share_elasticity ≈ -6.08`（p 值约 0.009），即对美关税相对巴西/阿根廷每上升 1 个单位，美国相对份额比显著下降。  
  - 在情景模拟中：
    - **Reciprocal tariff**：对美大豆关税从约 `0.23` 提高到 `0.48`，`q1_scenario_exports.csv` 中美国多期 `import_change_pct` 约为 `-7%` 左右；
    - **Full retaliation**：在此基础上再提高约 `0.5`，美国大豆进口量的降幅放大到约 `-14%` 左右，而巴西/阿根廷在样例数据下基本保持数量稳定、份额上升。

> 解释：Q1 的主结论是“在 Armington 来源替代框架下，中国对美大豆加征关税会显著压缩美国份额，并向巴西、阿根廷发生替代”，数值大小取决于最终采用的真实数据与参数。

---

### 3.2 Q2：日本汽车、产能外移与博弈均衡

- **赛题目标**  
  分析美国对日本整车提高关税后，日本车企在“继续出口—在美生产—在墨西哥生产”等路径间如何调整，以及这种调整如何影响美国汽车进口结构、本土产量与就业。

- **主要数据文件**  
  - `2025/data/processed/q2/q2_1.csv`：品牌维度的美国汽车市场结构与日本品牌份额（示例/模板数据）。

- **核心代码与函数**  
  文件：`2025/src/models/q2_autos.py`
  - 计量部分：`AutoTradeModel.estimate_import_structure_model()`、`estimate_industry_transmission_model()`，对应结果在 `results/q2/econometric/summary.json`。  
  - 多智能体博弈：`NashEquilibriumSolver` 与 `AutoTradeModel.run_marl_analysis()`，结果在 `results/q2/marl/drl_summary.json` 与 `nash_equilibrium.json`。  
  - Transformer 预测模块：`prepare_transformer_sequence_data()` 与 `build_transformer_model()`，结果在 `results/q2/transformer/training_results.json`。

- **情景与关键结果（当前样例数据）**  
  见 `results/q2/econometric/summary.json`：
  - 设定三种结构性情景：
    - `S0_no_response`：日本保持较高直接出口份额；
    - `S1_partial_relocation`：部分产能转移至美国；
    - `S2_aggressive_localization`：大幅在美国/墨西哥本地化生产。  
  - 在样例数据下，最大情景下美国本土生产和就业增幅约为 `≈ 8%` 量级，基线进口渗透率约 `13.8%`。

  见 `results/q2/marl/nash_equilibrium.json`：
  - 在离散策略格子上，找到的代表性纳什均衡为：
    - 美国关税：`us_tariff = 0.25`；
    - 日本应对：`japan_relocation = 1.0`（几乎完全通过本地化/第三地生产绕开关税）；
    - 对应收益：`us_payoff ≈ 4.06`、`japan_payoff ≈ 60.0`，且为帕累托最优点。

  见 `results/q2/transformer/training_results.json`：
  - 在极小样本（训练 16、测试 4）条件下，Transformer 对进口负担的预测取得 `r2 ≈ 0.70`，主要作为前沿方法展示。

> 解释：Q2 的算例说明，在模型设定下，美国提高对日关税在短期内有正向收益，而日本通过产能外移可以在很大程度上中和关税冲击，这削弱了关税作为长期产业保护工具的有效性。

---

### 3.3 Q3：半导体供应链效率–安全权衡

- **赛题目标**  
  在高、中、低端芯片分段的框架下，分析美国通过关税、补贴和出口管制等政策组合，对本国半导体自给率和对高风险供应方依赖度的影响，刻画“生产效率—供应安全”的权衡。

- **主要数据文件**  
  - `2025/data/processed/q3/q3_1.csv`：年度半导体主面板（产出、政策指数等）。
  - `2025/data/processed/q3/q3_2_supply_chain_segments.csv`：按 segment 的供应链结构与风险指标（由脚本生成）。
  - 结果文件：
    - `2025/results/q3/gnn/risk_analysis.json`；
    - `2025/results/q3_policy_scenarios.csv`；
    - `2025/results/q3_security_metrics.csv`。

- **核心代码与函数**  
  文件：`2025/src/models/q3_semiconductors.py`
  - 计量部分：`SemiconductorModel.estimate_trade_response()` 与 `estimate_output_response()`（分 segment 估计贸易与产出弹性）。
  - 安全指标：`SemiconductorModel.compute_security_metrics()`，生成自给率、中国依赖度与 `supply_risk_index`，结果写入 `q3_security_metrics.csv`。  
  - GNN/图分析：`SupplyChainGraph` 与 `SemiconductorModel.run_gnn_analysis()`，结果写入 `gnn/risk_analysis.json` 和 Markdown 报告。
  - 政策组合模拟：`SemiconductorModel.simulate_policy_combinations()`，结果写入 `q3_policy_scenarios.csv`。

- **关键结果（当前样例数据）**  
  见 `results/q3/gnn/risk_analysis.json`：
  - 基线供应链结构：节点包括 `China`、`Taiwan`、`South Korea`、`Japan`、`EU`、`USA` 共 6 个供应方；
  - 基线风险指标：
    - HHI 集中度约 `2150`（中度集中），
    - 安全指数 `security_index ≈ 57.8 / 100`，
    - 说明当前供应链在集中度与多元化之间处于中间状态。  
  - 中断情景（对中国、台湾、韩国分别施加 50%–100% 中断）显示：
    - `cascading_risk_score` 随中断强度单调上升；
    - 对高端供应方（如台湾）在相同中断率下的风险上升更快，说明其在网络中的关键性更高。

  见 `results/q3_policy_scenarios.csv`：
  - 对 `Policy_A_subsidy_only`（补贴为主）、`Policy_B_tariff_only`（关税为主）、`Policy_C_comprehensive`（综合政策）三种情景：
    - 三个 segment 的名义自给率在样例数据下均围绕 `25%` 变化极小（主要受占位数据影响）；
    - 成本指数从补贴情景的 `10` 上升到关税情景 `25`、综合情景 `35`；
    - 安全指数在补贴情景和综合情景略有提升，但综合情景的“安全收益/成本比”（`efficiency_security_ratio`）低于单纯补贴情景。

> 解释：当前算例强调“单靠关税提高半导体安全性效率较低”，相较之下，适度的补贴与出口管制组合在给定成本水平下可以带来更有利的安全改进。

---

### 3.4 Q4：关税收入 Laffer 曲线与第二任期情景

- **赛题目标**  
  构建以有效关税水平为核心解释变量的静态与动态 Laffer 曲线，分析关税上调对美国关税收入在短期与中期的影响，并预测特朗普第二任期不同关税路径下的累计收入变化。

- **主要数据文件**  
  - `2025/data/processed/q4/` 下的关税收入面板与情景设定文件（由数据处理脚本生成）。
  - 结果文件：
    - `2025/results/q4/econometric/static_laffer.json`；
    - `2025/results/q4/econometric/revenue_summary.json`；
    - `2025/results/q4/ml/model_comparison.json` 等。

- **核心代码与函数**  
  文件：`2025/src/models/q4_tariff_revenue.py`
  - `TariffRevenueModel.estimate_static_revenue_model()`：估计静态 Laffer 曲线，输出最优关税率等。  
  - `TariffRevenueModel.estimate_dynamic_import_response()`：估计进口额对关税变动的动态弹性。  
  - `TariffRevenueModel.simulate_second_term_revenue()`：在给定关税路径下模拟 2025–2029 年的关税收入。  
  - `TariffRevenueModel.train_ml_models()`、`train_arima_model()`：ML 增强模块，用于生成对比分布与敏感性分析。

- **关键结果（当前样例数据）**  
  见 `results/q4/econometric/static_laffer.json`：
  - 样例估计的静态 Laffer 曲线拟合度 `R² ≈ 0.80`，观测值仅 6 年；
  - 线性与二次项系数对应的最优平均关税率约为 `optimal_tariff_pct ≈ 14.3%`，说明在模型设定下存在明显“过高关税反而降低收入”的拐点。

  见 `results/q4/econometric/revenue_summary.json`：
  - 在给定的“政策情景”下，样例结果显示：
    - 基线累计收入约 `4.0×10^11`（美元量级）；
    - 政策情景累计收入约 `1.37×10^12`；
    - 二者差额约 `9.7×10^11`，显示在样例参数下提高关税在给定区间内仍处于 Laffer 曲线上升段。

  见 `results/q4/ml/model_comparison.json`：
  - ML 模型在同一情景下给出的累计收入明显低于计量模型（差距在 `-12%` 到 `-63%` 之间），提示在小样本情形下复杂模型对外推非常敏感，应主要作为敏感性分析工具。

> 解释：Q4 的静态/动态 Laffer 模型给出了“适度关税提升收入、过高关税压缩税基”的结构性解释；当前样例结果用于展示方法链条，最终数值应在接入真实数据后重新估计。

---

### 3.5 Q5：宏观、金融与制造业回流

- **赛题目标**  
  研究关税与报复措施通过哪些通道影响美国的 GDP 增长、工业生产、金融市场与制造业回流；评估互惠关税是否真的能在可接受代价下推动制造业回流。

- **主要数据文件**  
  - `2025/data/processed/q5/q5_4_integrated_panel.csv`：集成宏观、金融、关税指数与回流指标的年度面板数据。  
  - 结果文件：
    - `2025/results/q5/econometric/regressions.json`；
    - `2025/results/q5/econometric/reshoring_effects.json`；
    - `2025/results/q5/econometric/var_results.json`；
    - `2025/results/q5/ml/feature_importance.json` 等。

- **核心代码与函数**  
  文件：`2025/src/models/q5_macro_finance.py`
  - `MacroFinanceModel.estimate_regression_effects()`：对 GDP 增速、工业生产、制造业增加值占比等回归关税指数与报复指数。  
  - `MacroFinanceModel.estimate_var_model()`：构建包含关税指数、报复指数、GDP 增长与工业生产的 VAR 模型并生成冲击响应。  
  - `MacroFinanceModel.evaluate_reshoring()`：以 2025 年为事件点，对制造业回流进行前后对比与事件研究。  
  - `MacroFinanceModel.train_reshoring_ml()`：训练随机森林与梯度提升模型，输出制造业回流的特征重要性。

- **关键结果（当前样例数据）**  
  见 `results/q5/econometric/regressions.json`：
  - 关税与报复指数对宏观变量的回归 R² 较低，且多数系数在传统显著性水平下不显著，说明在当前样本和占位数据下，难以得到强统计结论；
  - 对制造业增加值占比（`manufacturing_va_share`），关税系数为负（约 `-0.19`），p 值约 `0.12`，方向上提示关税上升未必利于制造业占比提高。

  见 `results/q5/econometric/reshoring_effects.json`：
  - 事件研究结果中，处理前后制造业增加值占比的均值差约为 `-0.90` 百分点；
  - 回归估计的处理效应 `treatment_coef ≈ -0.79`，p 值约 `0.42`，**未达到统计显著**，说明在当前设定下难以证明互惠关税显著推动制造业回流。

  见 `results/q5/econometric/var_results.json`：
  - VAR(1) 模型在 10 年样本上给出的 AIC、BIC 有限，`is_stable` 标志为 `false`，提示模型在形式上略显不稳定；
  - 但关税冲击对 GDP 增长与工业生产的脉冲响应在数值上有限且逐步衰减，可用于**定性说明“关税与报复升级对宏观有一定负面压力”**。

  见 `results/q5/ml/feature_importance.json`：
  - 在针对制造业回流的 ML 模型中，
    - 关税总指数 `tariff_index_total` 与 `tariff_retaliation_interaction` 的重要性与金融指数 `sp500_index` 大致同量级；
    - 但在极小样本下，模型整体 `R²` 较低，应谨慎解读为“相关性参考”而非因果结论。

> 解释：Q5 的总体信息是：在当前占位数据与短样本下，很难统计上证明互惠关税显著改善宏观与制造业指标；更稳妥的用法是把回归与 VAR 结果视为机制说明，将 ML/VAR-LSTM 视为补充性的预测与敏感性工具。

---

## 4 运行与复现实验（简要）

完整的运行说明见 `2025/paper/APMCM_2025C_method_data_code_cn.md` 第 4 节，这里只给出简要思路：

1. **环境准备**  
   - 项目使用 Python 和常见科学计算/机器学习库（如 `pandas`、`numpy`、`statsmodels`、`scikit-learn`、`tensorflow/torch` 等）；
   - 具体依赖与环境管理方式（如 `uv`、`pyproject.toml` 等）以项目根目录实际文件和方法说明文档为准。

2. **数据准备**  
   - 若仅使用仓库中已生成的 `data/processed` 数据，可直接运行各题模型；
   - 如要接入真实的关税、贸易和宏观数据，应先运行相应的数据预处理脚本（位于 `2025/scripts`）以更新 `data/processed`，再重新运行模型。

3. **模型运行**  
   - 可在 Python 环境中直接调用各模型文件中的 `run_q1_analysis`、`run_q2_analysis`、`run_q3_analysis`、`run_q4_analysis`、`run_q5_analysis` 等入口函数（如在各文件中定义）；
   - 或参考 `2025/src/models/run_models.ps1` 中的示例，在 PowerShell 中批量运行对应脚本，统一生成 `results/q1`–`q5` 下的 JSON/CSV/Markdown 报告。

4. **结果检查**  
   - 每个子目录下的 `SUMMARY.md` 给出方法与文件结构总览；
   - 关键数值结论集中在 `*.json` 和情景表格 `*.csv` 中；
   - 图形通常保存在 `figures/`（如已生成）。

---

## 5 当前结果的适用范围与局限

- **数据层面**：
  - 仓库中部分外部数据为模板或占位数值，用于联调代码与演示分析框架；
  - 正式参赛或提交时，应替换为来自官方数据库（如 UN Comtrade、FRED、BEA、BLS、IMF、中国海关等）的真实数据，并重新跑全套模型。

- **计量与样本量**：
  - Q3–Q5 的年度样本量较小，多数模型更偏向结构解释与情景模拟，而非高精度预测；
  - Laffer 曲线与 VAR 模型的拟合结果需要结合经济直觉、敏感性分析和置信区间谨慎解读。

- **机器学习与深度学习模块**：
  - LSTM、Transformer、GNN、MARL 等模块在当前设定下更多承担“结构展示”和“前沿方法示例”的角色；
  - 在数据有限的情况下，其具体数值输出不宜直接用作政策结论，但可用来展示不同方法在同一问题上的视角差异。

---

## 6 快速索引：问题—代码—数据—结果

| 问题 | 模型文件 | 主要数据（processed） | 关键结果目录 |
|------|----------|------------------------|--------------|
| Q1 大豆 | `src/models/q1_soybeans.py` | `data/processed/q1/q1_1.csv` | `results/q1/`（`q1_elasticities.json`、`q1_scenario_exports.csv` 等） |
| Q2 汽车 | `src/models/q2_autos.py` | `data/processed/q2/q2_1.csv` | `results/q2/econometric/`、`results/q2/marl/`、`results/q2/transformer/` |
| Q3 半导体 | `src/models/q3_semiconductors.py` | `data/processed/q3/q3_1.csv` 等 | `results/q3/gnn/`、`results/q3/ml/`、`q3_policy_scenarios.csv`、`q3_security_metrics.csv` |
| Q4 收入 | `src/models/q4_tariff_revenue.py` | `data/processed/q4/` | `results/q4/econometric/`、`results/q4/ml/` |
| Q5 宏观与回流 | `src/models/q5_macro_finance.py` | `data/processed/q5/q5_4_integrated_panel.csv` 等 | `results/q5/econometric/`、`results/q5/ml/` |

本文件旨在把赛题、数据、代码和结果文件“串成一条线”，方便快速定位与交叉查阅；如需更详细的理论推导与文字表述，请结合 `2025/paper` 目录下的两份中文文稿使用。
