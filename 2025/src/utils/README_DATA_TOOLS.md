# 数据获取和处理工具使用指南

本目录包含用于APMCM 2025 Problem C的数据获取、清洗和质量检查工具。

---

## 工具概览

### 1. `data_fetch.py` - 官方数据源获取
从FRED和UN Comtrade等官方API获取数据

### 2. `worldbank_wits.py` - World Bank/WITS数据
获取World Bank指标和处理WITS批量下载文件

### 3. `data_consolidate.py` - 数据整合与质量检查
整合多个数据文件，执行质量检查

### 4. `china_soybeans_manual.py` - 中国大豆进口数据工具
处理手动下载的GACC数据，提供替代方案

---

## 快速开始

### 前置要求

项目使用uv进行包管理，确保已安装uv和Python 3.11+。

### API密钥配置

在项目根目录创建`.env`文件或设置环境变量：

```bash
# FRED API (必需，用于美国宏观数据)
FRED_API_KEY=your_fred_api_key

# UN Comtrade API (可选，当前不可用)
UN_COMTRADE_API_KEY=your_comtrade_key
```

**获取FRED API密钥**: https://fred.stlouisfed.org/docs/api/api_key.html

---

## 使用示例

### 1. 数据质量检查

检查所有外部数据文件的质量：

```bash
uv run python 2025/src/utils/data_consolidate.py check --data-dir 2025/data/external
```

输出包括：
- 文件行列数
- 缺失值统计
- 重复行检测
- 时间范围覆盖
- 质量评级

### 2. 下载FRED官方数据

#### 列出所有可用数据集
```bash
uv run python 2025/src/utils/data_fetch.py --list
```

#### 下载所有数据集
```bash
uv run python 2025/src/utils/data_fetch.py --datasets all
```

#### 下载特定问题组的数据
```bash
# Q5宏观经济数据
uv run python 2025/src/utils/data_fetch.py --groups Q5

# Q3半导体数据
uv run python 2025/src/utils/data_fetch.py --groups Q3

# Q2汽车数据
uv run python 2025/src/utils/data_fetch.py --groups Q2
```

#### 下载特定数据集
```bash
uv run python 2025/src/utils/data_fetch.py --datasets q5_real_gdp q5_cpi q5_unemployment_rate
```

### 3. 整合宏观数据

将多个FRED文件合并为单个宽表格式：

```bash
uv run python 2025/src/utils/data_consolidate.py merge-macro --data-dir 2025/data/external
```

生成文件: `us_macro_consolidated.csv`

包含列:
- year
- gdp_real (实际GDP)
- cpi (消费者价格指数)
- unemployment (失业率)
- industrial_production (工业生产指数)
- fed_funds_rate (联邦基金利率)
- treasury_10y (10年期国债收益率)
- sp500 (标普500指数)

### 4. 整合所有FRED数据

#### 长格式（每行一个观测值）
```bash
uv run python 2025/src/utils/data_consolidate.py consolidate --data-dir 2025/data/external --output fred_all.csv
```

#### 宽格式（每行一个年份，指标为列）
```bash
uv run python 2025/src/utils/data_consolidate.py consolidate --data-dir 2025/data/external --output fred_wide.csv --wide
```

### 5. World Bank数据获取

获取World Bank指标（例如：中国制造业关税率）：

```bash
uv run python 2025/src/utils/worldbank_wits.py wb TM.TAX.MANF.SM.AR.ZS \
    --country CHN \
    --start-year 2015 \
    --end-year 2024 \
    --output 2025/data/external/wb_china_tariff.csv
```

### 6. 处理WITS批量下载文件

如果从WITS网站手动下载了CSV：

```bash
uv run python 2025/src/utils/worldbank_wits.py wits \
    path/to/downloaded_wits_file.csv \
    --output 2025/data/external/wits_processed.csv \
    --year-hint Year \
    --partner-hint Partner \
    --value-hint "Trade Value"
```

---

## 中国大豆进口数据处理

由于UN Comtrade API当前不可用，使用以下工具处理手动下载的数据。

### 步骤1: 创建数据模板

生成一个模板文件，显示所需的数据结构：

```bash
uv run python 2025/src/utils/china_soybeans_manual.py template \
    --output 2025/data/raw/soybeans_template.csv
```

### 步骤2: 手动下载数据

**推荐数据源**:

1. **中国海关总署（GACC）**
   - URL: http://www.customs.gov.cn/
   - 导航: 数说海关 → 数据在线查询
   - 商品: HS 1201 (大豆)
   - 年份: 2015-2024
   - 按来源国分解

2. **UN Comtrade网页界面**
   - URL: https://comtradeplus.un.org/
   - Reporter: China (156)
   - Partner: All
   - Product: HS 1201
   - Flow: Imports
   - Years: 2015-2024

3. **WITS批量下载**
   - URL: https://wits.worldbank.org/
   - 需要注册账户
   - 提交Advanced Query并下载结果

### 步骤3: 处理下载的数据

假设从GACC下载了CSV文件：

```bash
uv run python 2025/src/utils/china_soybeans_manual.py process \
    2025/data/raw/gacc_soybeans_raw.csv \
    --output 2025/data/external/china_imports_soybeans_official.csv
```

工具会自动：
- 识别年份、国家、金额、数量列（支持中英文）
- 标准化国家名称（US/Brazil/Argentina）
- 计算贸易战期间的关税率
- 过滤到主要出口国

### 步骤4: 验证数据完整性

```bash
uv run python 2025/src/utils/china_soybeans_manual.py validate \
    2025/data/external/china_imports_soybeans_official.csv
```

检查内容：
- 年份覆盖范围（期望2015-2024）
- 出口国完整性（期望US、Brazil、Argentina）
- 缺失值统计
- 异常值检测

---

## 常见问题

### Q: FRED API请求失败怎么办？

**A**: 检查以下几点：
1. API密钥是否正确设置
2. 是否超过API调用限制（每天1000次）
3. 网络连接是否正常

可以添加`--max-retries`和`--retry-sleep`参数：
```bash
uv run python 2025/src/utils/data_fetch.py --datasets q5_real_gdp
```

### Q: UN Comtrade API为什么一直失败？

**A**: UN Comtrade Plus API自2023年以来多次调整认证机制。当前版本可能需要：
- 重新注册并获取新的API密钥
- 使用不同的认证头格式
- 直接使用网页界面手动下载

**推荐方案**: 使用`china_soybeans_manual.py`处理手动下载的数据。

### Q: World Bank数据格式看起来很奇怪？

**A**: 旧版本的工具存储了嵌套字典。已在新版本修复。重新运行：
```bash
uv run python 2025/src/utils/worldbank_wits.py wb <indicator> --country <code>
```

新版本会自动清洗并展开嵌套字典。

### Q: 如何查看某个FRED系列的详细信息？

**A**: 访问 https://fred.stlouisfed.org/series/<SERIES_ID>

例如: https://fred.stlouisfed.org/series/GDPC1

### Q: 数据整合后如何在模型中使用？

**A**: 在分析脚本中：
```python
import pandas as pd

# 使用整合的宏观数据
macro_df = pd.read_csv('2025/data/external/us_macro_consolidated.csv')

# 或使用单个指标
gdp_df = pd.read_csv('2025/data/external/us_real_gdp_official.csv')
```

---

## 数据文件命名规范

- `*_official.csv` - 从官方API直接获取的数据
- `*_consolidated.csv` - 整合后的多指标数据集
- `*_template.csv` - 用于手动数据输入的模板
- `*_processed.csv` - 处理过的手动下载数据

---

## 输出目录结构

```
2025/data/
├── external/           # 最终使用的外部数据
│   ├── *_official.csv  # 官方API数据
│   ├── us_macro_consolidated.csv
│   └── failed_downloads.jsonl  # 失败记录
├── raw/               # 原始下载文件
│   ├── gacc_*.csv
│   └── *_template.csv
├── interim/           # 中间处理结果
└── processed/         # 最终处理后的分析数据
```

---

## 技术细节

### data_fetch.py

**支持的数据源**:
- FRED (Federal Reserve Economic Data)
- UN Comtrade Plus API (当前不可用)

**重试机制**:
- 默认3次重试
- 指数退避策略
- 记录失败到`failed_downloads.jsonl`

### worldbank_wits.py

**依赖**:
- `wbdata` library (可选)
- pandas

**数据清洗**:
- 自动展开嵌套字典列
- 标准化列名
- 类型转换

### data_consolidate.py

**质量检查指标**:
- 缺失值百分比
- 重复行数量
- 时间覆盖范围
- 列完整性

**质量评级**:
- EXCELLENT: 无缺失值，无重复
- GOOD: <5%缺失值，无重复
- NEEDS REVIEW: ≥5%缺失或有重复

### china_soybeans_manual.py

**列名匹配**:
- 支持中英文列名
- 模糊匹配（不区分大小写）
- 自动检测年份/国家/金额/数量列

**关税计算**:
- 正常MFN税率: 3%
- 2018-2020美国贸易战期间: 25%
- 可在输出后手动调整

---

## 开发和调试

### 启用详细日志

在脚本开头添加：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

或在命令行中设置环境变量：
```bash
export PYTHONLOGLEVEL=DEBUG
uv run python 2025/src/utils/data_fetch.py --datasets q5_real_gdp
```

### 测试单个函数

```python
from src.utils.data_fetch import fetch_fred_series

# 测试获取单个系列
df = fetch_fred_series(
    'GDPC1',
    start_year=2020,
    end_year=2024,
    frequency='a',
    output_filename='test_gdp.csv'
)
print(df.head())
```

---

## 更新日志

### 2025-11-20
- ✅ 修复`data_fetch.py`重复定义bug
- ✅ 改进`worldbank_wits.py`数据清洗
- ✅ 新增`data_consolidate.py`整合工具
- ✅ 新增`china_soybeans_manual.py`替代方案
- ✅ 生成完整数据质量报告

---

## 支持和反馈

如有问题或建议，请查看：
- 数据质量报告: `project_document/data_quality_report_20251120.md`
- 数据计划: `project_document/data_plan_status_20251120_1242.md`

---

**最后更新**: 2025-11-20 20:00
