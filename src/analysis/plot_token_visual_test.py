import hydra
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from transformers import AutoTokenizer
from typing import Dict, List
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

class TokenPerformanceVisualizer:
    """Create clean scatter plots of token count vs performance"""
    
    def __init__(self, results_dir: str, dataset_path: str, model_configs: Dict):
        self.results_dir = Path(results_dir)
        self.dataset_path = dataset_path
        self.model_configs = model_configs
        self.tokenizers = {}
        self._load_tokenizers()
        self._load_dataset()
        self._load_results()
        self._merge_token_counts()
    
    def _load_tokenizers(self):
        """Load tokenizers for each model"""
        logger.info("Loading tokenizers...")
        for model_name, tokenizer_name in self.model_configs.items():
            logger.info(f"  Loading {model_name}: {tokenizer_name}")
            try:
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                    tokenizer_name, 
                    trust_remote_code=True
                )
            except Exception as e:
                logger.error(f"Failed to load tokenizer for {model_name}: {e}")
                continue
    
    def _load_dataset(self):
        """Load dataset and compute token counts"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
        
        self.token_counts = {}
        for model_name, tokenizer in self.tokenizers.items():
            logger.info(f"Computing token counts for {model_name}...")
            model_tokens = []
            
            for sample in self.dataset:
                words = sample['words']
                word_list_text = ', '.join(words)
                tokens = tokenizer.encode(word_list_text, add_special_tokens=False)
                
                model_tokens.append({
                    'id': sample['id'],
                    'seed': sample['seed'],
                    'pattern': sample['pattern'],
                    'count': sample['count'],
                    'trial': sample['trial'],
                    'token_count': len(tokens),
                    'model': model_name
                })
            
            self.token_counts[model_name] = pd.DataFrame(model_tokens)
    
    def _load_results(self):
        """Load all result CSV files"""
        logger.info(f"Loading results from {self.results_dir}")
        
        csv_files = list(self.results_dir.glob("results_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No results files found in {self.results_dir}")
        
        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            if 'model' not in df.columns:
                parts = csv_file.stem.split('_')
                if len(parts) >= 3:
                    df['model'] = '_'.join(parts[2:])
            dfs.append(df)
        
        self.results = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(self.results)} result samples")
    
    def _merge_token_counts(self):
        """Merge token counts with results"""
        all_tokens = pd.concat(self.token_counts.values(), ignore_index=True)
        
        self.merged = self.results.merge(
            all_tokens,
            on=['id', 'model', 'seed', 'pattern', 'count', 'trial'],
            how='left'
        )
        
        # Filter to only 'c' and 'r' patterns
        self.merged = self.merged[self.merged['pattern'].isin(['c', 'r'])].copy()
        
        logger.info(f"Merged dataset: {len(self.merged)} samples (c and r only)")
    
    def plot_model_scatter(self, model: str, rule_count: int, output_path: Path):
        """
        Create scatter plot for one model and one rule count
        
        X-axis: Token count
        Y-axis: Score (0-1)
        Color: Blue for coherent (c), Red for random (r)
        """
        data = self.merged[
            (self.merged['model'] == model) & 
            (self.merged['count'] == rule_count)
        ].dropna(subset=['token_count', 'score'])
        
        if len(data) == 0:
            logger.warning(f"No data for {model} with {rule_count} rules")
            return
        
        # Set up plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Define colors
        colors = {
            'c': '#2E86DE',  # Blue for coherent
            'r': '#EE5A6F'   # Red for random
        }
        
        pattern_names = {
            'c': 'Coherent',
            'r': 'Random'
        }
        
        # Plot each pattern
        for pattern in ['c', 'r']:
            pattern_data = data[data['pattern'] == pattern]
            
            if len(pattern_data) == 0:
                continue
            
            ax.scatter(
                pattern_data['token_count'],
                pattern_data['score'],
                c=colors[pattern],
                label=pattern_names[pattern],
                alpha=0.6,
                s=80,
                edgecolors='black',
                linewidth=0.8
            )
            
            # Add trend line for each pattern
            if len(pattern_data) > 2:
                z = np.polyfit(pattern_data['token_count'], pattern_data['score'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(
                    pattern_data['token_count'].min(),
                    pattern_data['token_count'].max(),
                    100
                )
                ax.plot(x_line, p(x_line), 
                       color=colors[pattern], 
                       linestyle='--', 
                       linewidth=2.5, 
                       alpha=0.8)
                
                # Calculate correlation for this pattern
                corr, p_val = stats.pearsonr(pattern_data['token_count'], pattern_data['score'])
                logger.info(f"  {pattern_names[pattern]}: r={corr:.3f}, p={p_val:.3e}, n={len(pattern_data)}")
        
        # Styling
        ax.set_xlabel('Total Input Tokens', fontsize=14, fontweight='bold')
        ax.set_ylabel('Rules Followed (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'{model.upper()} - {rule_count} Rules', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set y-axis to percentage (0-100%)
        ax.set_ylim(-5, 105)
        y_ticks = np.arange(0, 101, 20)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{int(y)}%' for y in y_ticks])
        
        # Scale scores to percentage for display
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.2)])
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
        ax.set_axisbelow(True)
        
        # Legend
        legend = ax.legend(loc='best', fontsize=12, frameon=True, 
                          shadow=True, fancybox=True)
        legend.get_frame().set_alpha(0.9)
        
        # Add statistics box
        overall_corr, overall_p = stats.pearsonr(data['token_count'], data['score'])
        stats_text = f'Overall: r={overall_corr:.3f}\np={overall_p:.3e}'
        ax.text(0.02, 0.98, stats_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved: {output_path}")
    
    def plot_model_all_rules(self, model: str, output_path: Path):
        """
        Create multi-panel plot showing all rule counts for one model
        """
        data = self.merged[self.merged['model'] == model].dropna(subset=['token_count', 'score'])
        
        if len(data) == 0:
            logger.warning(f"No data for {model}")
            return
        
        rule_counts = sorted(data['count'].unique())
        n_rules = len(rule_counts)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_rules, figsize=(7*n_rules, 6))
        if n_rules == 1:
            axes = [axes]
        
        # Define colors
        colors = {
            'c': '#2E86DE',  # Blue for coherent
            'r': '#EE5A6F'   # Red for random
        }
        
        pattern_names = {
            'c': 'Coherent',
            'r': 'Random'
        }
        
        for idx, rule_count in enumerate(rule_counts):
            ax = axes[idx]
            rule_data = data[data['count'] == rule_count]
            
            # Plot each pattern
            for pattern in ['c', 'r']:
                pattern_data = rule_data[rule_data['pattern'] == pattern]
                
                if len(pattern_data) == 0:
                    continue
                
                ax.scatter(
                    pattern_data['token_count'],
                    pattern_data['score'],
                    c=colors[pattern],
                    label=pattern_names[pattern],
                    alpha=0.6,
                    s=60,
                    edgecolors='black',
                    linewidth=0.8
                )
                
                # Trend line
                if len(pattern_data) > 2:
                    z = np.polyfit(pattern_data['token_count'], pattern_data['score'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(
                        pattern_data['token_count'].min(),
                        pattern_data['token_count'].max(),
                        100
                    )
                    ax.plot(x_line, p(x_line), 
                           color=colors[pattern], 
                           linestyle='--', 
                           linewidth=2, 
                           alpha=0.8)
            
            # Styling
            ax.set_xlabel('Total Input Tokens', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Rules Followed (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{rule_count} Rules', fontsize=13, fontweight='bold')
            
            # Y-axis as percentage
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.2)])
            
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            
            # Legend only on first subplot
            if idx == 0:
                ax.legend(loc='best', fontsize=10, frameon=True)
            
            # Add correlation
            overall_corr, overall_p = stats.pearsonr(rule_data['token_count'], rule_data['score'])
            stats_text = f'r={overall_corr:.3f}'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.suptitle(f'{model.upper()} - Token Count vs Performance',
                    fontsize=15, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved: {output_path}")
    
    def plot_all_models_one_rule(self, rule_count: int, output_path: Path):
        """
        Create multi-panel plot showing all models for one rule count
        """
        data = self.merged[self.merged['count'] == rule_count].dropna(subset=['token_count', 'score'])
        
        if len(data) == 0:
            logger.warning(f"No data for {rule_count} rules")
            return
        
        models = sorted(data['model'].unique())
        n_models = len(models)
        
        # Calculate grid dimensions
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        # Define colors
        colors = {
            'c': '#2E86DE',  # Blue for coherent
            'r': '#EE5A6F'   # Red for random
        }
        
        pattern_names = {
            'c': 'Coherent',
            'r': 'Random'
        }
        
        for idx, model in enumerate(models):
            ax = axes[idx]
            model_data = data[data['model'] == model]
            
            # Plot each pattern
            for pattern in ['c', 'r']:
                pattern_data = model_data[model_data['pattern'] == pattern]
                
                if len(pattern_data) == 0:
                    continue
                
                ax.scatter(
                    pattern_data['token_count'],
                    pattern_data['score'],
                    c=colors[pattern],
                    label=pattern_names[pattern],
                    alpha=0.6,
                    s=60,
                    edgecolors='black',
                    linewidth=0.8
                )
                
                # Trend line
                if len(pattern_data) > 2:
                    z = np.polyfit(pattern_data['token_count'], pattern_data['score'], 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(
                        pattern_data['token_count'].min(),
                        pattern_data['token_count'].max(),
                        100
                    )
                    ax.plot(x_line, p(x_line), 
                           color=colors[pattern], 
                           linestyle='--', 
                           linewidth=2, 
                           alpha=0.8)
            
            # Styling
            ax.set_xlabel('Total Input Tokens', fontsize=11, fontweight='bold')
            ax.set_ylabel('Rules Followed (%)', fontsize=11, fontweight='bold')
            ax.set_title(f'{model.upper()}', fontsize=12, fontweight='bold')
            
            # Y-axis as percentage
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_yticklabels([f'{int(y*100)}%' for y in np.arange(0, 1.1, 0.2)])
            
            ax.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(loc='best', fontsize=9, frameon=True)
            
            # Add correlation
            overall_corr, overall_p = stats.pearsonr(model_data['token_count'], model_data['score'])
            stats_text = f'r={overall_corr:.3f}'
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'All Models - {rule_count} Rules',
                    fontsize=15, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved: {output_path}")
    
    def generate_all_plots(self, output_dir: Path):
        """Generate all scatter plot variations"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*70}")
        logger.info("Generating Token vs Performance Scatter Plots")
        logger.info("(Coherent vs Random only)")
        logger.info(f"{'='*70}\n")
        
        models = sorted(self.merged['model'].unique())
        rule_counts = sorted(self.merged['count'].unique())
        
        # 1. Individual plots: one per model per rule count
        individual_dir = output_dir / "individual"
        individual_dir.mkdir(exist_ok=True)
        
        logger.info("📊 Generating individual plots (model × rule count)...")
        for model in models:
            for rule_count in rule_counts:
                logger.info(f"  {model} - {rule_count} rules")
                output_path = individual_dir / f"scatter_{model}_{rule_count}rules.png"
                self.plot_model_scatter(model, rule_count, output_path)
        
        # 2. Multi-panel: all rule counts for each model
        multipanel_dir = output_dir / "by_model"
        multipanel_dir.mkdir(exist_ok=True)
        
        logger.info("\n📊 Generating multi-panel plots (by model)...")
        for model in models:
            logger.info(f"  {model}")
            output_path = multipanel_dir / f"scatter_{model}_all_rules.png"
            self.plot_model_all_rules(model, output_path)
        
        # 3. Multi-panel: all models for each rule count
        by_rules_dir = output_dir / "by_rules"
        by_rules_dir.mkdir(exist_ok=True)
        
        logger.info("\n📊 Generating multi-panel plots (by rule count)...")
        for rule_count in rule_counts:
            logger.info(f"  {rule_count} rules")
            output_path = by_rules_dir / f"scatter_all_models_{rule_count}rules.png"
            self.plot_all_models_one_rule(rule_count, output_path)
        
        logger.info(f"\n{'='*70}")
        logger.info("🎉 All plots generated!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info(f"{'='*70}")


def get_available_models(conf_path: str = "conf/model") -> List[str]:
    """Get list of available model configs"""
    model_dir = Path(conf_path)
    models = [f.stem for f in model_dir.glob("*.yaml")]
    return sorted(models)


def load_model_configs(model_names: List[str]) -> Dict[str, str]:
    """Load model configurations and extract tokenizer names"""
    model_configs = {}
    
    for model_name in model_names:
        model_cfg_path = Path(f"conf/model/{model_name}.yaml")
        
        if not model_cfg_path.exists():
            logger.warning(f"Model config not found: {model_cfg_path}, skipping...")
            continue
        
        with open(model_cfg_path) as f:
            model_cfg = OmegaConf.load(f)
        
        tokenizer_name = model_cfg.model.name
        model_configs[model_name] = tokenizer_name
    
    return model_configs