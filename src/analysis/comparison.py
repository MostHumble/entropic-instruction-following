import pandas as pd
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

class MultiModelComparison:
    """Compare results across multiple models"""
    
    def __init__(self, results_dir: str = "data/results"):
        self.results_dir = Path(results_dir)
        self.all_results = self._load_all_results()
    
    def _load_all_results(self) -> pd.DataFrame:
        """Load and combine all results CSV files"""
        results_files = list(self.results_dir.glob("results_*.csv"))
        
        if not results_files:
            raise FileNotFoundError(f"No results CSV files found in {self.results_dir}")
        
        dfs = []
        for rf in sorted(results_files):
            df = pd.read_csv(rf)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        return combined_df
    
    def plot_model_comparison(self):
        """Plot average score by model"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall scores
        model_scores = self.all_results.groupby('model')['score'].agg(['mean', 'std']).reset_index()
        model_scores = model_scores.sort_values('mean', ascending=False)
        
        axes[0].bar(range(len(model_scores)), model_scores['mean'], yerr=model_scores['std'], 
                    color='steelblue', alpha=0.7, capsize=5)
        axes[0].set_xticks(range(len(model_scores)))
        axes[0].set_xticklabels(model_scores['model'], rotation=45, ha='right')
        axes[0].set_ylabel("Mean Score")
        axes[0].set_title("Model Performance Comparison")
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Scores by pattern
        for model in self.all_results['model'].unique():
            model_data = self.all_results[self.all_results['model'] == model]
            pattern_scores = model_data.groupby('pattern')['score'].mean()
            axes[1].plot(pattern_scores.index, pattern_scores.values, marker='o', label=model)
        
        axes[1].set_xlabel("Pattern")
        axes[1].set_ylabel("Mean Score")
        axes[1].set_title("Performance by Pattern")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison.png", dpi=300)
        plt.close()
    
    def get_summary(self) -> str:
        """Get text summary of model comparison"""
        summary = "\n" + "="*70 + "\nMODEL COMPARISON\n" + "="*70 + "\n"
        
        for model in sorted(self.all_results['model'].unique()):
            model_data = self.all_results[self.all_results['model'] == model]
            mean_score = model_data['score'].mean()
            std_score = model_data['score'].std()
            n_samples = len(model_data)
            
            summary += f"\n{model}:\n"
            summary += f"  Mean score: {mean_score:.2%} (±{std_score:.4f})\n"
            summary += f"  Samples: {n_samples}\n"
        
        return summary