"""
Unified Data Export Interface for All Models.

This module provides a standardized way to export model results in multiple formats:
- JSON: Structured data with metadata
- CSV: Tabular data for analysis
- Markdown: Human-readable reports

All exports follow the directory structure:
2025/results/q{N}/{method}/
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataExporter:
    """Unified data exporter for all model results."""
    
    def __init__(self, results_base: Path):
        """Initialize exporter with base results directory.
        
        Args:
            results_base: Base directory for results (e.g., RESULTS_DIR / 'q2')
        """
        self.results_base = Path(results_base)
        self.results_base.mkdir(parents=True, exist_ok=True)
        
    def export_json(self, data: Dict[str, Any], method: str, filename: str) -> Path:
        """Export data as JSON.
        
        Args:
            data: Dictionary to export
            method: Method name (e.g., 'econometric', 'ml', 'transformer')
            filename: Output filename (without extension)
            
        Returns:
            Path to exported file
        """
        method_dir = self.results_base / method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        export_data = {
            'metadata': {
                'exported_at': datetime.now().isoformat(),
                'method': method,
                'question': self.results_base.name
            },
            'data': data
        }
        
        output_path = method_dir / f"{filename}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported JSON: {output_path}")
        return output_path
    
    def export_csv(self, data: Union[pd.DataFrame, List[Dict]], method: str, filename: str) -> Path:
        """Export data as CSV.
        
        Args:
            data: DataFrame or list of dicts to export
            method: Method name
            filename: Output filename (without extension)
            
        Returns:
            Path to exported file
        """
        method_dir = self.results_base / method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        output_path = method_dir / f"{filename}.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Exported CSV: {output_path}")
        return output_path
    
    def export_markdown(self, content: Union[str, Dict], method: str, filename: str) -> Path:
        """Export data as Markdown report.
        
        Args:
            content: String content or dict to format as markdown
            method: Method name
            filename: Output filename (without extension)
            
        Returns:
            Path to exported file
        """
        method_dir = self.results_base / method
        method_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(content, dict):
            # Format dict as markdown
            lines = [
                f"# {filename.replace('_', ' ').title()}",
                "",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Method:** {method}",
                "",
                "## Results",
                "",
                "```json",
                json.dumps(content, indent=2, ensure_ascii=False),
                "```"
            ]
            content = "\n".join(lines)
        
        output_path = method_dir / f"{filename}.md"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Exported Markdown: {output_path}")
        return output_path
    
    def export_all(self, data: Dict[str, Any], method: str, base_filename: str,
                   include_csv: bool = True, include_md: bool = True) -> Dict[str, Path]:
        """Export data in all formats.
        
        Args:
            data: Data to export
            method: Method name
            base_filename: Base filename (without extension)
            include_csv: Whether to export CSV (if data is tabular)
            include_md: Whether to export Markdown
            
        Returns:
            Dict mapping format to output path
        """
        outputs = {}
        
        # Always export JSON
        outputs['json'] = self.export_json(data, method, base_filename)
        
        # Export CSV if data is tabular
        if include_csv and 'results' in data and isinstance(data['results'], (list, pd.DataFrame)):
            outputs['csv'] = self.export_csv(data['results'], method, base_filename)
        
        # Export Markdown
        if include_md:
            outputs['markdown'] = self.export_markdown(data, method, base_filename)
        
        return outputs
    
    def create_summary_report(self, methods: List[str], title: str) -> Path:
        """Create a summary report across all methods.
        
        Args:
            methods: List of method names to include
            title: Report title
            
        Returns:
            Path to summary report
        """
        lines = [
            f"# {title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Methods Overview",
            ""
        ]
        
        for method in methods:
            method_dir = self.results_base / method
            if method_dir.exists():
                files = list(method_dir.glob('*.json'))
                lines.extend([
                    f"### {method.title()}",
                    f"- **Directory:** `{method_dir}`",
                    f"- **Files:** {len(files)}",
                    ""
                ])
                
                # List key files
                for file in sorted(files)[:5]:  # Show first 5 files
                    lines.append(f"  - `{file.name}`")
                
                if len(files) > 5:
                    lines.append(f"  - ... and {len(files) - 5} more files")
                
                lines.append("")
        
        lines.extend([
            "## Data Structure",
            "",
            "```",
            f"{self.results_base.name}/",
        ])
        
        for method in methods:
            method_dir = self.results_base / method
            if method_dir.exists():
                lines.append(f"├── {method}/")
                files = sorted(method_dir.glob('*'))[:3]
                for i, file in enumerate(files):
                    prefix = "│   ├──" if i < len(files) - 1 else "│   └──"
                    lines.append(f"{prefix} {file.name}")
                if len(list(method_dir.glob('*'))) > 3:
                    lines.append("│   └── ...")
        
        lines.extend([
            "```",
            "",
            "## Usage",
            "",
            "All results are saved in three formats:",
            "- **JSON**: Machine-readable with metadata",
            "- **CSV**: Tabular data for analysis",
            "- **Markdown**: Human-readable reports",
            ""
        ])
        
        output_path = self.results_base / 'SUMMARY.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Created summary report: {output_path}")
        return output_path


class ModelResultsManager:
    """Manager for organizing and exporting model results."""
    
    def __init__(self, question_number: int, results_base_dir: Path):
        """Initialize results manager.
        
        Args:
            question_number: Question number (2, 3, 4, or 5)
            results_base_dir: Base results directory (e.g., RESULTS_DIR)
        """
        self.question_number = question_number
        self.results_dir = results_base_dir / f'q{question_number}'
        self.exporter = DataExporter(self.results_dir)
        self.methods = []
        
    def register_method(self, method_name: str) -> Path:
        """Register a new method and create its directory.
        
        Args:
            method_name: Name of the method (e.g., 'econometric', 'ml')
            
        Returns:
            Path to method directory
        """
        if method_name not in self.methods:
            self.methods.append(method_name)
        
        method_dir = self.results_dir / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Registered method: {method_name} at {method_dir}")
        
        return method_dir
    
    def save_results(self, method: str, results: Dict[str, Any], 
                    name: str, formats: List[str] = ['json', 'csv', 'md']) -> Dict[str, Path]:
        """Save results in specified formats.
        
        Args:
            method: Method name
            results: Results data
            name: Base filename
            formats: List of formats to export ('json', 'csv', 'md')
            
        Returns:
            Dict mapping format to file path
        """
        outputs = {}
        
        if 'json' in formats:
            outputs['json'] = self.exporter.export_json(results, method, name)
        
        if 'csv' in formats and 'results' in results:
            try:
                outputs['csv'] = self.exporter.export_csv(results['results'], method, name)
            except Exception as e:
                logger.warning(f"Could not export CSV: {e}")
        
        if 'md' in formats:
            outputs['md'] = self.exporter.export_markdown(results, method, name)
        
        return outputs
    
    def generate_summary(self) -> Path:
        """Generate summary report for this question.
        
        Returns:
            Path to summary report
        """
        title = f"Q{self.question_number} Results Summary"
        return self.exporter.create_summary_report(self.methods, title)
    
    def get_method_dir(self, method: str) -> Path:
        """Get directory path for a method.
        
        Args:
            method: Method name
            
        Returns:
            Path to method directory
        """
        return self.results_dir / method
    
    def list_all_results(self) -> Dict[str, List[str]]:
        """List all result files by method.
        
        Returns:
            Dict mapping method name to list of filenames
        """
        all_results = {}
        
        for method in self.methods:
            method_dir = self.results_dir / method
            if method_dir.exists():
                files = [f.name for f in method_dir.glob('*') if f.is_file()]
                all_results[method] = sorted(files)
        
        return all_results


# Convenience functions for common export patterns

def export_model_metrics(manager: ModelResultsManager, method: str, 
                        metrics: Dict[str, float], model_name: str) -> None:
    """Export model performance metrics.
    
    Args:
        manager: Results manager
        method: Method name
        metrics: Dictionary of metrics (e.g., {'rmse': 0.5, 'r2': 0.8})
        model_name: Name of the model
    """
    results = {
        'model': model_name,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    manager.save_results(method, results, f'{model_name}_metrics', formats=['json', 'md'])


def export_predictions(manager: ModelResultsManager, method: str,
                      predictions: pd.DataFrame, model_name: str) -> None:
    """Export model predictions.
    
    Args:
        manager: Results manager
        method: Method name
        predictions: DataFrame with predictions
        model_name: Name of the model
    """
    results = {
        'model': model_name,
        'results': predictions,
        'timestamp': datetime.now().isoformat(),
        'num_predictions': len(predictions)
    }
    
    manager.save_results(method, results, f'{model_name}_predictions', formats=['json', 'csv'])


def export_comparison(manager: ModelResultsManager, 
                     comparisons: Dict[str, Dict[str, float]],
                     title: str = "Model Comparison") -> None:
    """Export model comparison results.
    
    Args:
        manager: Results manager
        comparisons: Dict mapping model name to metrics
        title: Comparison title
    """
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparisons).T
    comparison_df.index.name = 'model'
    comparison_df = comparison_df.reset_index()
    
    results = {
        'title': title,
        'results': comparison_df,
        'timestamp': datetime.now().isoformat(),
        'best_model': comparison_df.loc[comparison_df['r2'].idxmax(), 'model'] if 'r2' in comparison_df.columns else None
    }
    
    manager.save_results('comparison', results, 'model_comparison', formats=['json', 'csv', 'md'])
