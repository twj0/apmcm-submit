"""
Q6综合面板数据集生成脚本
整合Q1-Q5处理后的关键指标，构建用于综合分析的面板数据集
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.config import DATA_PROCESSED

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_q1_data() -> pd.DataFrame:
    """加载Q1大豆贸易数据"""
    try:
        # 加载年度数据
        annual_file = DATA_PROCESSED / 'q1' / 'q1_0.csv'
        if annual_file.exists():
            q1_annual = pd.read_csv(annual_file)
            logger.info(f"Loaded Q1 annual data: {len(q1_annual)} rows")
        else:
            logger.warning(f"Q1 annual data file not found: {annual_file}")
            q1_annual = pd.DataFrame()
            
        # 加载月度数据
        monthly_file = DATA_PROCESSED / 'q1' / 'q1_1.csv'
        if monthly_file.exists():
            q1_monthly = pd.read_csv(monthly_file)
            # 聚合为年度数据
            q1_monthly_agg = q1_monthly.groupby('year').agg({
                'price_index': 'mean',
                'trade_volume_kmt': 'sum'
            }).reset_index()
            logger.info(f"Loaded and aggregated Q1 monthly data: {len(q1_monthly_agg)} rows")
        else:
            logger.warning(f"Q1 monthly data file not found: {monthly_file}")
            q1_monthly_agg = pd.DataFrame()
            
        # 合并数据
        if not q1_annual.empty and not q1_monthly_agg.empty:
            q1_data = pd.merge(q1_annual, q1_monthly_agg, on='year', how='outer')
        elif not q1_annual.empty:
            q1_data = q1_annual
        elif not q1_monthly_agg.empty:
            q1_data = q1_monthly_agg
        else:
            q1_data = pd.DataFrame(columns=['year'])
            
        return q1_data
    except Exception as e:
        logger.error(f"Error loading Q1 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q2_data() -> pd.DataFrame:
    """加载Q2汽车产业数据"""
    try:
        # 加载品牌销量数据
        sales_file = DATA_PROCESSED / 'q2' / 'q2_0_us_auto_sales_by_brand.csv'
        if sales_file.exists():
            q2_sales = pd.read_csv(sales_file)
            # 聚合为年度数据
            q2_sales_agg = q2_sales.groupby('year')['sales_units'].sum().reset_index()
            q2_sales_agg.rename(columns={'sales_units': 'auto_sales_total'}, inplace=True)
            logger.info(f"Loaded and aggregated Q2 sales data: {len(q2_sales_agg)} rows")
        else:
            logger.warning(f"Q2 sales data file not found: {sales_file}")
            q2_sales_agg = pd.DataFrame()
            
        # 加载行业指标数据
        indicators_file = DATA_PROCESSED / 'q2' / 'q2_1_industry_indicators.csv'
        if indicators_file.exists():
            q2_indicators = pd.read_csv(indicators_file)
            logger.info(f"Loaded Q2 industry indicators: {len(q2_indicators)} rows")
        else:
            logger.warning(f"Q2 industry indicators file not found: {indicators_file}")
            q2_indicators = pd.DataFrame()
            
        # 合并数据
        if not q2_sales_agg.empty and not q2_indicators.empty:
            q2_data = pd.merge(q2_sales_agg, q2_indicators, on='year', how='outer')
        elif not q2_sales_agg.empty:
            q2_data = q2_sales_agg
        elif not q2_indicators.empty:
            q2_data = q2_indicators
        else:
            q2_data = pd.DataFrame(columns=['year'])
            
        return q2_data
    except Exception as e:
        logger.error(f"Error loading Q2 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q3_data() -> pd.DataFrame:
    """加载Q3半导体数据"""
    try:
        # 加载半导体产出数据
        output_file = DATA_PROCESSED / 'q3' / 'q3_0_us_semiconductor_output.csv'
        if output_file.exists():
            q3_output = pd.read_csv(output_file)
            logger.info(f"Loaded Q3 output data: {len(q3_output)} rows")
        else:
            logger.warning(f"Q3 output data file not found: {output_file}")
            q3_output = pd.DataFrame()
            
        # 加载安全指标数据
        security_file = DATA_PROCESSED / 'q3' / 'q3_security_metrics.csv'
        if security_file.exists():
            q3_security = pd.read_csv(security_file)
            logger.info(f"Loaded Q3 security metrics: {len(q3_security)} rows")
        else:
            logger.warning(f"Q3 security metrics file not found: {security_file}")
            q3_security = pd.DataFrame()
            
        # 合并数据
        if not q3_output.empty and not q3_security.empty:
            q3_data = pd.merge(q3_output, q3_security, on='year', how='outer')
        elif not q3_output.empty:
            q3_data = q3_output
        elif not q3_security.empty:
            q3_data = q3_security
        else:
            q3_data = pd.DataFrame(columns=['year'])
            
        return q3_data
    except Exception as e:
        logger.error(f"Error loading Q3 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q4_data() -> pd.DataFrame:
    """加载Q4关税收入数据"""
    try:
        # 加载关税收入面板数据
        revenue_file = DATA_PROCESSED / 'q4' / 'q4_0_tariff_revenue_panel.csv'
        if revenue_file.exists():
            q4_revenue = pd.read_csv(revenue_file)
            logger.info(f"Loaded Q4 revenue data: {len(q4_revenue)} rows")
        else:
            logger.warning(f"Q4 revenue data file not found: {revenue_file}")
            q4_revenue = pd.DataFrame()
            
        return q4_revenue
    except Exception as e:
        logger.error(f"Error loading Q4 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q5_data() -> pd.DataFrame:
    """加载Q5宏观金融数据"""
    try:
        data_frames = []
        
        # 加载宏观指标
        macro_file = DATA_PROCESSED / 'q5' / 'q5_0_macro_indicators.csv'
        if macro_file.exists():
            q5_macro = pd.read_csv(macro_file)
            data_frames.append(q5_macro)
            logger.info(f"Loaded Q5 macro indicators: {len(q5_macro)} rows")
        else:
            logger.warning(f"Q5 macro indicators file not found: {macro_file}")
            
        # 加载金融指标
        finance_file = DATA_PROCESSED / 'q5' / 'q5_1_financial_indicators.csv'
        if finance_file.exists():
            q5_finance = pd.read_csv(finance_file)
            data_frames.append(q5_finance)
            logger.info(f"Loaded Q5 financial indicators: {len(q5_finance)} rows")
        else:
            logger.warning(f"Q5 financial indicators file not found: {finance_file}")
            
        # 加载回流指标
        reshoring_file = DATA_PROCESSED / 'q5' / 'q5_2_reshoring_indicators.csv'
        if reshoring_file.exists():
            q5_reshoring = pd.read_csv(reshoring_file)
            data_frames.append(q5_reshoring)
            logger.info(f"Loaded Q5 reshoring indicators: {len(q5_reshoring)} rows")
        else:
            logger.warning(f"Q5 reshoring indicators file not found: {reshoring_file}")
            
        # 加载报复指数
        retaliation_file = DATA_PROCESSED / 'q5' / 'q5_3_retaliation_index.csv'
        if retaliation_file.exists():
            q5_retaliation = pd.read_csv(retaliation_file)
            data_frames.append(q5_retaliation)
            logger.info(f"Loaded Q5 retaliation index: {len(q5_retaliation)} rows")
        else:
            logger.warning(f"Q5 retaliation index file not found: {retaliation_file}")
            
        # 合并所有Q5数据
        if data_frames:
            q5_data = data_frames[0]
            for df in data_frames[1:]:
                q5_data = pd.merge(q5_data, df, on='year', how='outer')
        else:
            q5_data = pd.DataFrame(columns=['year'])
            
        return q5_data
    except Exception as e:
        logger.error(f"Error loading Q5 data: {e}")
        return pd.DataFrame(columns=['year'])

def create_q6_dataset(output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    创建Q6综合面板数据集
    
    Args:
        output_path: 输出文件路径，如果提供则保存为CSV文件
        
    Returns:
        pd.DataFrame: Q6综合面板数据集
    """
    logger.info("Starting Q6 dataset creation...")
    
    # 加载各问题数据
    q1_data = load_q1_data()
    q2_data = load_q2_data()
    q3_data = load_q3_data()
    q4_data = load_q4_data()
    q5_data = load_q5_data()
    
    # 初始化主数据框（以年份为主键）
    years = set()
    for df in [q1_data, q2_data, q3_data, q4_data, q5_data]:
        if 'year' in df.columns:
            years.update(df['year'].unique())
    
    if not years:
        logger.error("No year data found in any dataset")
        return pd.DataFrame()
    
    # 创建基础年份框架
    q6_dataset = pd.DataFrame({'year': sorted(list(years))})
    
    # 依次合并各问题数据
    datasets = [
        ('Q1', q1_data),
        ('Q2', q2_data),
        ('Q3', q3_data),
        ('Q4', q4_data),
        ('Q5', q5_data)
    ]
    
    for name, data in datasets:
        if not data.empty and 'year' in data.columns:
            q6_dataset = pd.merge(q6_dataset, data, on='year', how='left')
            logger.info(f"Merged {name} data: {len(data)} rows")
        else:
            logger.warning(f"Skipping {name} data - empty or missing year column")
    
    # 数据类型优化
    for col in q6_dataset.columns:
        if col != 'year':
            # 尝试转换为数值类型
            q6_dataset[col] = pd.to_numeric(q6_dataset[col], errors='ignore')
    
    # 排序
    q6_dataset = q6_dataset.sort_values('year').reset_index(drop=True)
    
    # 保存到文件（如果指定了路径）
    if output_path:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            q6_dataset.to_csv(output_path, index=False)
            logger.info(f"Q6 dataset saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving Q6 dataset: {e}")
    
    logger.info(f"Q6 dataset creation completed: {len(q6_dataset)} rows, {len(q6_dataset.columns)} columns")
    return q6_dataset

if __name__ == "__main__":
    # 创建Q6数据集并保存到默认位置
    output_file = DATA_PROCESSED / 'q6' / 'q6_0_final_integrated_dataset.csv'
    q6_df = create_q6_dataset(output_file)
    
    # 显示数据集基本信息
    if not q6_df.empty:
        print("\nQ6 Dataset Summary:")
        print(f"Shape: {q6_df.shape}")
        print(f"Year range: {q6_df['year'].min()} - {q6_df['year'].max()}")
        print("\nColumns:")
        for col in q6_df.columns:
            print(f"  - {col}")
        print("\nFirst 5 rows:")
        print(q6_df.head())
    else:
        print("Failed to create Q6 dataset")