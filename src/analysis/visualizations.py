import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict

class ResultsVisualizer:
    """Centralized visualization for experiment results"""
    
    def __init__(self, results_csv: str, output_dir: str = "data/results"):
        self.df = pd.read_csv(results_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._expand_word_details()
        self._setup_color_scheme()
    
    def _setup_color_scheme(self):
        """Dynamically generate colors for all patterns"""
        patterns = sorted(self.df['pattern'].unique())
        
        # Use matplotlib's qualitative color palettes
        if len(patterns) <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif len(patterns) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            # For many patterns, use a continuous colormap
            colors = plt.cm.hsv(np.linspace(0, 0.9, len(patterns)))
        
        self.color_map = {pattern: colors[i] for i, pattern in enumerate(patterns)}
        
        # Override with semantic colors if pattern type is obvious
        semantic_colors = {
            'c': '#2ca02c',      # Green for coherent
            'r': '#d62728',      # Red for random
        }
        
        for pattern in patterns:
            # If pattern is just 'c' or 'r', use semantic color
            if pattern in semantic_colors:
                self.color_map[pattern] = semantic_colors[pattern]
            # If pattern starts with 'c' and has no 'r', make it greenish
            elif pattern.startswith('c') and 'r' not in pattern.lower():
                # Vary green shades
                self.color_map[pattern] = plt.cm.Greens(0.5 + 0.3 * (hash(pattern) % 5) / 5)
            # If pattern starts with 'r' and has no 'c', make it reddish
            elif pattern.startswith('r') and 'c' not in pattern.lower():
                self.color_map[pattern] = plt.cm.Reds(0.5 + 0.3 * (hash(pattern) % 5) / 5)
    
    def _get_component_colors(self, component_types: list) -> Dict[str, str]:
        """Dynamically assign colors to component types"""
        colors = {}
        for ct in component_types:
            if ct.startswith('c'):
                colors[ct] = '#2ca02c'  # Green
            elif ct.startswith('r'):
                colors[ct] = '#d62728'  # Red
            else:
                colors[ct] = '#ff7f0e'  # Orange for unknown
        return colors
    
    def _expand_word_details(self):
        """Expand word_details JSON into separate rows for analysis"""
        expanded_rows = []
        
        for _, row in self.df.iterrows():
            # Extract trial info from ID
            trial_id = int(row['id'].split('_')[-2])
            config_id = '_'.join(row['id'].split('_')[:-2])
            
            word_details = json.loads(row['word_details'])
            for wd in word_details:
                expanded_rows.append({
                    'pattern': row['pattern'],
                    'count': row['count'], 
                    'score': row['score'],
                    'position_in_rule': wd['position'],
                    'word': wd['word'],
                    'found': wd['found'],
                    'positions_in_text': wd['positions_in_text'],
                    'occurrences': wd['occurrences'],
                    'sample_id': row['id'],
                    'trial_id': trial_id,
                    'config_id': config_id
                })
        
        self.expanded_df = pd.DataFrame(expanded_rows)
    
    def plot_follow_rate_by_position_absolute(self):
        """Separate subplots for each pattern and count combination"""
        grouped = self.expanded_df.groupby(['pattern', 'count'])
        n_groups = len(grouped)
        
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), sharex=False, sharey=True)
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, ((pattern, count), group_data) in enumerate(grouped):
            ax = axes[idx]
            
            max_position = group_data['position_in_rule'].max()
            
            position_stats = group_data.groupby(['position_in_rule', 'trial_id'])['found'].mean().reset_index()
            position_agg = position_stats.groupby('position_in_rule').agg({
                'found': ['mean', 'sem', 'count']
            }).reset_index()
            
            position_agg.columns = ['position', 'mean', 'sem', 'count']
            
            # Use dynamic color
            color = self.color_map.get(pattern, 'steelblue')
            
            ax.plot(position_agg['position'], position_agg['mean'], 
                   marker='o', linewidth=2, markersize=3, color=color)
            
            ax.fill_between(
                position_agg['position'],
                position_agg['mean'] - 1.96 * position_agg['sem'],
                position_agg['mean'] + 1.96 * position_agg['sem'],
                alpha=0.3, color=color
            )
            
            n_trials = int(position_agg['count'].iloc[0])
            ax.set_title(f"Pattern: {pattern}, Count: {count} (n={n_trials} trials)", 
                        fontsize=11, fontweight='bold')
            ax.set_xlabel("Absolute Position in Rule", fontsize=10)
            ax.set_ylabel("Follow Rate", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            overall_mean = position_agg['mean'].mean()
            ax.axhline(y=overall_mean, color='red', linestyle='-', linewidth=1.5, alpha=0.7,
                      label=f'Mean: {overall_mean:.2%}')
            ax.legend(fontsize=8)
            
            ax.set_xlim(-2, max_position + 2)
        
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Follow Rate by Absolute Position (Separated by Pattern and Count)", 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / "01a_follow_rate_by_absolute_position.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_follow_rate_heatmap_absolute(self):
        """Heatmap with absolute positions, NORMALIZED by rule length"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        n_counts = len(rule_counts)
        fig, axes = plt.subplots(n_counts, 1, figsize=(16, 4*n_counts))
        if n_counts == 1:
            axes = [axes]
        
        for idx, count in enumerate(rule_counts):
            ax = axes[idx]
            count_data = self.expanded_df[self.expanded_df['count'] == count].copy()
            
            max_pos = count_data['position_in_rule'].max()
            count_data['position_pct'] = (count_data['position_in_rule'] / max_pos * 100).astype(int)
            
            trial_agg = count_data.groupby(['pattern', 'position_pct', 'trial_id'])['found'].mean().reset_index()
            final_agg = trial_agg.groupby(['pattern', 'position_pct'])['found'].mean().reset_index()
            
            heatmap_data = final_agg.pivot(index='pattern', columns='position_pct', values='found')
            
            sns.heatmap(
                heatmap_data, 
                cmap='RdYlGn', 
                cbar_kws={'label': 'Follow Rate'},
                ax=ax,
                vmin=0,
                vmax=1,
                linewidths=0.5,
                linecolor='white',
                annot=False,
                xticklabels=10
            )
            
            ax.set_xlabel("Position in Rule (%)", fontsize=12)
            ax.set_ylabel("Pattern Type", fontsize=12)
            ax.set_title(f"Follow Rate Heatmap (Rule Length: {count} words)", 
                        fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01b_follow_rate_heatmap_absolute.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pattern_performance_overview(self):
        """Bar plot: Overall pattern performance BY RULE COUNT"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        pattern_count_scores = self.expanded_df.groupby(['pattern', 'count'])['found'].agg(
            mean='mean',      # Use named aggregations
            std='std',
            n='count'        
        ).reset_index()
        
        patterns = sorted(pattern_count_scores['pattern'].unique())
        counts = sorted(pattern_count_scores['count'].unique())
        
        x = np.arange(len(patterns))
        width = 0.8 / len(counts) if len(counts) > 1 else 0.6
        
        # Generate colors for count bars
        count_colors = plt.cm.viridis(np.linspace(0, 0.8, len(counts)))
        
        for idx, count in enumerate(counts):
            count_data = pattern_count_scores[pattern_count_scores['count'] == count]
            count_data = count_data.set_index('pattern').reindex(patterns).reset_index()
            
            offset = (idx - len(counts)/2 + 0.5) * width
            ax.bar(x + offset, count_data['mean'], width,
                   label=f'{count} words', alpha=0.8,
                   yerr=count_data['std'], capsize=3,
                   color=count_colors[idx])
        
        ax.set_xlabel("Pattern Type", fontsize=12)
        ax.set_ylabel("Follow Rate", fontsize=12)
        ax.set_title("Overall Follow Rate by Pattern and Rule Length", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend(title="Rule Length", loc='best')
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01d_pattern_performance_by_count.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_rule_position_vs_text_position(self):
        """Scatter plots: Rule position vs position in text, separated by count"""
        # Group by both pattern and count
        grouped = self.expanded_df.groupby(['pattern', 'count'])
        n_groups = len(grouped)
        
        n_cols = 3
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_groups == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, ((pattern, count), group_data) in enumerate(grouped):
            ax = axes[idx]
            
            # Filter to only found words with positions
            pattern_data = group_data[
                (group_data['found'] == True) &
                (group_data['occurrences'] > 0)
            ].copy()
            
            if len(pattern_data) == 0:
                ax.text(0.5, 0.5, f"No data\nPattern: {pattern}\nCount: {count}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{pattern} ({count} words)")
                continue
            
            pattern_data['first_pos_in_text'] = pattern_data['positions_in_text'].apply(
                lambda x: x[0] if x and len(x) > 0 else -1
            )
            pattern_data = pattern_data[pattern_data['first_pos_in_text'] >= 0]
            
            if len(pattern_data) == 0:
                ax.text(0.5, 0.5, f"No valid positions\nPattern: {pattern}\nCount: {count}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{pattern} ({count} words)")
                continue
            
            # Use pattern color
            color = self.color_map.get(pattern, 'steelblue')
            
            ax.scatter(
                pattern_data['position_in_rule'], 
                pattern_data['first_pos_in_text'],
                alpha=0.5, 
                s=50,
                c=[color] * len(pattern_data),
                edgecolors='black',
                linewidths=0.5
            )
            
            # Add trend line if enough data
            if len(pattern_data) > 2:
                z = np.polyfit(pattern_data['position_in_rule'], 
                              pattern_data['first_pos_in_text'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(pattern_data['position_in_rule'].min(), 
                                     pattern_data['position_in_rule'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                       label=f'Trend (slope={z[0]:.2f})')
                
                # Calculate correlation
                corr = np.corrcoef(pattern_data['position_in_rule'], 
                                  pattern_data['first_pos_in_text'])[0, 1]
                ax.text(0.05, 0.95, f'r={corr:.3f}', 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel("Position in Rule", fontsize=10)
            ax.set_ylabel("Character Position in Text", fontsize=10)
            ax.set_title(f"Pattern: {pattern}, Count: {count}", fontsize=11, fontweight='bold')
            if len(pattern_data) > 2:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_groups, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Rule Position vs Text Position (Separated by Pattern and Count)", 
                    fontsize=15, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_rule_position_vs_text_position.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        self.plot_follow_rate_by_position_absolute()
        print(" Follow rate by absolute position (separate subplots)")
        
        self.plot_follow_rate_heatmap_absolute()
        print(" Follow rate heatmap (absolute positions)")
                        
        self.plot_pattern_performance_overview()
        print(" Pattern performance overview")
        
        self.plot_rule_position_vs_text_position()
        print(" Rule position vs text position")
        
        print(f" All visualizations saved to {self.output_dir}")