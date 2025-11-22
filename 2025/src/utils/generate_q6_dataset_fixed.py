#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q6综合面板数据集生成脚本

整合Q1-Q5的关键指标，构建用于综合分析的面板数据集。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 项目根目录路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

def load_q1_data() -> pd.DataFrame:
    """加载Q1大豆贸易数据"""
    try:
        # 加载年度数据 (实际文件是q1_1.csv)
        annual_file = DATA_PROCESSED / 'q1' / 'q1_1.csv'
        if annual_file.exists():
            q1_annual = pd.read_csv(annual_file)
            # 提取年份（从period字段）
            if 'period' in q1_annual.columns:
                q1_annual['year'] = (q1_annual['period'] / 100).astype(int)
                # 聚合年度数据
                agg_dict = {}
                for col in q1_annual.columns:
                    if col in ['net_weight_tons', 'primary_value_usd']:
                        agg_dict[col] = 'sum'
                    elif col in ['quality_grade', 'tariff_rate']:
                        agg_dict[col] = 'mean'
                
                q1_annual = q1_annual.groupby('year').agg(agg_dict).reset_index()
                logger.info(f"Loaded and aggregated Q1 annual data: {len(q1_annual)} rows")
            else:
                logger.warning("Period column not found in Q1 data")
                q1_annual = pd.DataFrame()
        else:
            logger.warning(f"Q1 annual data file not found: {annual_file}")
            q1_annual = pd.DataFrame()
            
        # Q1没有月度数据文件，所以这部分为空
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
        # 加载品牌销量数据 (实际文件是q2_1.csv)
        sales_file = DATA_PROCESSED / 'q2' / 'q2_1.csv'
        if sales_file.exists():
            q2_data = pd.read_csv(sales_file)
            logger.info(f"Loaded Q2 data: {len(q2_data)} rows")
        else:
            logger.warning(f"Q2 data file not found: {sales_file}")
            q2_data = pd.DataFrame()
            
        return q2_data
    except Exception as e:
        logger.error(f"Error loading Q2 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q3_data() -> pd.DataFrame:
    """加载Q3半导体数据"""
    try:
        # 加载半导体产出数据 (实际文件是q3_1.csv)
        output_file = DATA_PROCESSED / 'q3' / 'q3_1.csv'
        if output_file.exists():
            q3_data = pd.read_csv(output_file)
            logger.info(f"Loaded Q3 data: {len(q3_data)} rows")
        else:
            logger.warning(f"Q3 data file not found: {output_file}")
            q3_data = pd.DataFrame()
            
        return q3_data
    except Exception as e:
        logger.error(f"Error loading Q3 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q4_data() -> pd.DataFrame:
    """加载Q4关税收入数据"""
    try:
        # 加载关税收入面板数据 (实际文件是q4_1.csv)
        revenue_file = DATA_PROCESSED / 'q4' / 'q4_1.csv'
        if revenue_file.exists():
            q4_data = pd.read_csv(revenue_file)
            # 只保留历史数据（scenario == 'historical'）
            if 'scenario' in q4_data.columns:
                q4_data = q4_data[q4_data['scenario'] == 'historical'].drop('scenario', axis=1)
            logger.info(f"Loaded Q4 data: {len(q4_data)} rows")
        else:
            logger.warning(f"Q4 data file not found: {revenue_file}")
            q4_data = pd.DataFrame()
            
        return q4_data
    except Exception as e:
        logger.error(f"Error loading Q4 data: {e}")
        return pd.DataFrame(columns=['year'])

def load_q5_data() -> pd.DataFrame:
    """加载Q5宏观金融数据"""
    try:
        # 加载宏观金融数据 (实际文件是q5_1.csv)
        macro_file = DATA_PROCESSED / 'q5' / 'q5_1.csv'
        if macro_file.exists():
            q5_data = pd.read_csv(macro_file)
            logger.info(f"Loaded Q5 data: {len(q5_data)} rows")
        else:
            logger.warning(f"Q5 data file not found: {macro_file}")
            q5_data = pd.DataFrame()
            
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