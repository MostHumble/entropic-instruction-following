import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from pathlib import Path

class ResultsVisualizer:
    """Centralized visualization for experiment results"""
    
    def __init__(self, results_csv: str, output_dir: str = "data/results"):
        self.df = pd.read_csv(results_csv)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._expand_word_details()
    
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
        """Separate subplots for each pattern - NO overlapping!"""
        patterns = sorted(self.expanded_df['pattern'].unique())
        n_patterns = len(patterns)
        
        # Create subplots in a grid
        n_cols = 3
        n_rows = (n_patterns + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), sharex=True, sharey=True)
        axes = axes.flatten() if n_patterns > 1 else [axes]
        
        max_position = self.expanded_df['position_in_rule'].max()
        
        for idx, pattern in enumerate(patterns):
            ax = axes[idx]
            pattern_data = self.expanded_df[self.expanded_df['pattern'] == pattern]
            
            # Aggregate across trials, keep absolute position
            position_stats = pattern_data.groupby(['position_in_rule', 'trial_id'])['found'].mean().reset_index()
            position_agg = position_stats.groupby('position_in_rule').agg({
                'found': ['mean', 'sem', 'count']
            }).reset_index()
            
            position_agg.columns = ['position', 'mean', 'sem', 'count']
            
            # Plot with 95% CI
            ax.plot(position_agg['position'], position_agg['mean'], 
                   marker='o', linewidth=2, markersize=3, color='steelblue')
            
            ax.fill_between(
                position_agg['position'],
                position_agg['mean'] - 1.96 * position_agg['sem'],
                position_agg['mean'] + 1.96 * position_agg['sem'],
                alpha=0.3, color='steelblue'
            )
            
            ax.set_title(f"Pattern: {pattern} (n={int(position_agg['count'].iloc[0])} trials)", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel("Absolute Position in Rule", fontsize=10)
            ax.set_ylabel("Follow Rate", fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            # Add mean line
            overall_mean = position_agg['mean'].mean()
            ax.axhline(y=overall_mean, color='red', linestyle='-', linewidth=1.5, alpha=0.7,
                      label=f'Mean: {overall_mean:.2%}')
            ax.legend(fontsize=9)
        
        # Hide unused subplots
        for idx in range(n_patterns, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f"Follow Rate by Absolute Position (Rule Length: {max_position+1} words)", 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / "01a_follow_rate_by_absolute_position.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_follow_rate_heatmap_absolute(self):
        """Heatmap with absolute positions (no binning)"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Aggregate by trial, then average
        trial_agg = self.expanded_df.groupby(['pattern', 'position_in_rule', 'trial_id'])['found'].mean().reset_index()
        final_agg = trial_agg.groupby(['pattern', 'position_in_rule'])['found'].mean().reset_index()
        
        # Pivot for heatmap
        heatmap_data = final_agg.pivot(index='pattern', columns='position_in_rule', values='found')
        
        sns.heatmap(
            heatmap_data, 
            cmap='RdYlGn', 
            cbar_kws={'label': 'Follow Rate (averaged across trials)'},
            ax=ax,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='white',
            annot=False,
            xticklabels=10  # Show every 10th position to avoid clutter
        )
        
        ax.set_xlabel("Absolute Position in Rule", fontsize=12)
        ax.set_ylabel("Pattern Type", fontsize=12)
        ax.set_title("Follow Rate Heatmap (Absolute Positions, Aggregated Across Trials)", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01b_follow_rate_heatmap_absolute.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_follow_rate_comparison_clean(self):
        """Clean comparison: One line per pattern, different colors, clear legend"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        patterns = sorted(self.expanded_df['pattern'].unique())
        
        # Color scheme: coherent=green, random=red, mixed=orange/purple
        color_map = {
            'c': '#2ca02c',      # Green
            'r': '#d62728',      # Red
            'cr': '#ff7f0e',     # Orange
            'c|r': '#9467bd',    # Purple
            'r|c|r': '#8c564b',  # Brown
            'c|r|c': '#e377c2',  # Pink
            'c|r|c|r': '#7f7f7f' # Gray
        }
        
        for pattern in patterns:
            pattern_data = self.expanded_df[self.expanded_df['pattern'] == pattern]
            
            # Aggregate across trials
            position_stats = pattern_data.groupby(['position_in_rule', 'trial_id'])['found'].mean().reset_index()
            position_agg = position_stats.groupby('position_in_rule').agg({
                'found': ['mean', 'sem', 'count']
            }).reset_index()
            
            position_agg.columns = ['position', 'mean', 'sem', 'count']
            
            # Use every 5th point to reduce clutter
            step = max(1, len(position_agg) // 40)
            plot_data = position_agg.iloc[::step]
            
            color = color_map.get(pattern, '#1f77b4')
            n_trials = int(position_agg['count'].iloc[0])
            
            ax.plot(plot_data['position'], plot_data['mean'], 
                   marker='o', linewidth=2, markersize=4, 
                   color=color, alpha=0.8,
                   label=f"{pattern} (n={n_trials})")
        
        ax.set_xlabel("Absolute Position in Rule", fontsize=12)
        ax.set_ylabel("Follow Rate", fontsize=12)
        ax.set_title("Follow Rate Comparison Across Patterns", fontsize=14, fontweight='bold')
        ax.legend(title="Pattern", loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01c_follow_rate_comparison_clean.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_follow_rate_by_pattern_and_position(self):
        """Combined view with percentiles (original functionality)"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        max_position = self.expanded_df['position_in_rule'].max()
        rule_length = max_position + 1
        
        ax1 = axes[0]
        
        for pattern in sorted(self.expanded_df['pattern'].unique()):
            pattern_data = self.expanded_df[self.expanded_df['pattern'] == pattern]
            
            # Aggregate across trials first, then positions
            position_stats = pattern_data.groupby(['position_in_rule', 'trial_id'])['found'].mean().reset_index()
            position_agg = position_stats.groupby('position_in_rule').agg({
                'found': ['mean', 'sem', 'count']
            }).reset_index()
            
            position_agg.columns = ['position_in_rule', 'mean', 'sem', 'count']
            position_agg['position_pct'] = (position_agg['position_in_rule'] / max_position) * 100
            
            # Sample points to reduce clutter
            step = max(1, len(position_agg) // 40)
            plot_data = position_agg.iloc[::step]
            
            ax1.plot(plot_data['position_pct'], plot_data['mean'], 
                    marker='o', label=f"{pattern} (n={int(plot_data['count'].iloc[0])} trials)", 
                    linewidth=2, markersize=4, alpha=0.8)
            
            ax1.fill_between(
                plot_data['position_pct'],
                plot_data['mean'] - 1.96 * plot_data['sem'],
                plot_data['mean'] + 1.96 * plot_data['sem'],
                alpha=0.2
            )
        
        ax1.set_xlabel("Position in Rule (%)", fontsize=12)
        ax1.set_ylabel("Follow Rate", fontsize=12)
        ax1.set_title(f"Follow Rate by Pattern (Rule Length: {rule_length} words, with 95% CI)", 
                     fontsize=14, fontweight='bold')
        ax1.legend(title="Pattern", loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Heatmap (binned into deciles)
        ax2 = axes[1]
        
        trial_agg = self.expanded_df.groupby(['pattern', 'position_in_rule', 'trial_id'])['found'].mean().reset_index()
        final_agg = trial_agg.groupby(['pattern', 'position_in_rule'])['found'].mean().reset_index()
        
        final_agg['position_bin'] = pd.cut(
            final_agg['position_in_rule'], 
            bins=10, 
            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)]
        )
        
        heatmap_data = final_agg.groupby(['pattern', 'position_bin'])['found'].mean().unstack()
        
        sns.heatmap(
            heatmap_data, 
            cmap='RdYlGn', 
            cbar_kws={'label': 'Follow Rate'},
            ax=ax2,
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='white',
            annot=False
        )
        
        ax2.set_xlabel("Position in Rule (Deciles)", fontsize=12)
        ax2.set_ylabel("Pattern Type", fontsize=12)
        ax2.set_title("Follow Rate Heatmap (Aggregated Across Trials)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_follow_rate_by_position_pattern.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pattern_performance_overview(self):
        """Bar plot: Overall pattern performance with position breakdown"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Overall follow rate by pattern
        ax1 = axes[0]
        pattern_scores = self.expanded_df.groupby('pattern')['found'].agg(['mean', 'std', 'count']).reset_index()
        pattern_scores = pattern_scores.sort_values('mean', ascending=True)
        
        colors = ['#d62728' if p == 'r' else '#2ca02c' if p == 'c' else '#ff7f0e' 
                  for p in pattern_scores['pattern']]
        
        bars = ax1.barh(pattern_scores['pattern'], pattern_scores['mean'], 
                        xerr=pattern_scores['std'], color=colors, alpha=0.7, capsize=5)
        
        ax1.set_xlabel("Follow Rate", fontsize=12)
        ax1.set_ylabel("Pattern Type", fontsize=12)
        ax1.set_title("Overall Follow Rate by Pattern", fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (pattern, mean) in enumerate(zip(pattern_scores['pattern'], pattern_scores['mean'])):
            ax1.text(mean + 0.02, i, f'{mean:.1%}', va='center', fontsize=10, fontweight='bold')
        
        # Plot 2: Stacked area showing position-based contribution
        ax2 = axes[1]
        
        # Bin positions into quintiles
        self.expanded_df['position_quintile'] = pd.qcut(
            self.expanded_df['position_in_rule'], 
            q=5, 
            duplicates='drop',
            labels=['1st 20%', '2nd 20%', '3rd 20%', '4th 20%', '5th 20%']
        )
        
        quintile_data = self.expanded_df.groupby(['pattern', 'position_quintile'])['found'].mean().unstack()
        quintile_data.plot(kind='bar', ax=ax2, stacked=False, alpha=0.8, width=0.8)
        
        ax2.set_xlabel("Pattern Type", fontsize=12)
        ax2.set_ylabel("Follow Rate", fontsize=12)
        ax2.set_title("Follow Rate by Position Quintile", fontsize=14, fontweight='bold')
        ax2.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
        ax2.set_ylim(0, 1)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01b_pattern_performance_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_rule_position_vs_text_position(self):
        """Scatter plots: Rule position vs position in text"""
        patterns = sorted(self.expanded_df['pattern'].unique())
        n_patterns = len(patterns)
        n_cols = 3
        n_rows = (n_patterns + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_patterns > 1 else [axes]
        
        for idx, pattern in enumerate(patterns):
            ax = axes[idx]
            pattern_data = self.expanded_df[
                (self.expanded_df['pattern'] == pattern) & 
                (self.expanded_df['found'] == True) &
                (self.expanded_df['occurrences'] > 0)
            ].copy()
            
            if len(pattern_data) == 0:
                ax.text(0.5, 0.5, f"No data for pattern: {pattern}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Pattern: {pattern}")
                continue
            
            pattern_data['first_pos_in_text'] = pattern_data['positions_in_text'].apply(
                lambda x: x[0] if x else -1
            )
            pattern_data = pattern_data[pattern_data['first_pos_in_text'] >= 0]
            
            if len(pattern_data) == 0:
                ax.text(0.5, 0.5, f"No valid positions for: {pattern}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Pattern: {pattern}")
                continue
            
            scatter = ax.scatter(
                pattern_data['position_in_rule'], 
                pattern_data['first_pos_in_text'],
                alpha=0.5, 
                s=50,
                c=pattern_data['position_in_rule'],
                cmap='viridis',
                edgecolors='black',
                linewidths=0.5
            )
            
            if len(pattern_data) > 2:
                z = np.polyfit(pattern_data['position_in_rule'], 
                              pattern_data['first_pos_in_text'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(pattern_data['position_in_rule'].min(), 
                                     pattern_data['position_in_rule'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                       label=f'Trend (slope={z[0]:.2f})')
            
            ax.set_xlabel("Position in Rule", fontsize=10)
            ax.set_ylabel("Character Position in Text", fontsize=10)
            ax.set_title(f"Pattern: {pattern}", fontsize=12, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_patterns, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_rule_position_vs_text_position.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_follow_rate_by_component_type(self):
        """Bar plot: Follow rate by component type (coherent vs random)"""
        import re
        
        def get_component_type(row):
            pattern = row['pattern']
            position = row['position_in_rule']
            
            if '|' in pattern:
                components = pattern.split('|')
            else:
                components = re.findall(r'[cr]\d*', pattern)
            
            if len(components) == 0:
                return 'unknown'
            
            component = components[position % len(components)]
            return component
        
        self.expanded_df['component_type'] = self.expanded_df.apply(get_component_type, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        component_stats = self.expanded_df.groupby('component_type')['found'].agg(
            ['sum', 'count', 'mean', 'std']
        ).reset_index()
        component_stats = component_stats.sort_values('mean', ascending=False)
        
        colors = ['#2ca02c' if ct.startswith('c') else '#d62728' if ct.startswith('r') else '#ff7f0e'
                  for ct in component_stats['component_type']]
        
        bars = ax.barh(component_stats['component_type'], component_stats['mean'], 
                       color=colors, alpha=0.7)
        
        ax.set_xlabel("Follow Rate", fontsize=12)
        ax.set_ylabel("Component Type", fontsize=12)
        ax.set_title("Follow Rate by Component Type (Coherent vs Random)", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (ct, mean, count) in enumerate(zip(component_stats['component_type'], 
                                                    component_stats['mean'],
                                                    component_stats['count'])):
            ax.text(mean + 0.02, i, f'{mean:.1%} (n={int(count)})', 
                   va='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_follow_rate_by_component_type.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_primacy_recency_bias(self):
        """Line plots: Primacy & Recency bias analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        self.expanded_df['position_quintile'] = pd.qcut(
            self.expanded_df['position_in_rule'], 
            q=5, 
            duplicates='drop', 
            labels=['1st', '2nd', '3rd', '4th', '5th']
        )
        
        quintile_follow = self.expanded_df.groupby('position_quintile')['found'].agg(
            ['sum', 'count', 'mean', 'std']
        ).reset_index()
        quintile_follow['follow_rate'] = quintile_follow['mean']
        
        axes[0].bar(range(len(quintile_follow)), quintile_follow['follow_rate'], 
                   color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(len(quintile_follow)))
        axes[0].set_xticklabels(quintile_follow['position_quintile'])
        axes[0].set_ylabel("Follow Rate", fontsize=12)
        axes[0].set_xlabel("Position Quintile", fontsize=12)
        axes[0].set_title("Primacy & Recency Effect", fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 1)
        axes[0].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, rate in enumerate(quintile_follow['follow_rate']):
            axes[0].text(i, rate + 0.03, f'{rate:.1%}', ha='center', fontweight='bold')
        
        pattern_quintile = self.expanded_df.groupby(['pattern', 'position_quintile'])['found'].mean().unstack()
        pattern_quintile.plot(kind='bar', ax=axes[1], alpha=0.8, width=0.8, edgecolor='black')
        axes[1].set_ylabel("Follow Rate", fontsize=12)
        axes[1].set_xlabel("Pattern Type", fontsize=12)
        axes[1].set_title("Follow Rate by Position Quintile (per Pattern)", fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1)
        axes[1].legend(title='Position Quintile', bbox_to_anchor=(1.05, 1))
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_primacy_recency_bias.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        # New plots (absolute positions, no overlap)
        self.plot_follow_rate_by_position_absolute()
        print("✅ Follow rate by absolute position (separate subplots)")
        
        self.plot_follow_rate_heatmap_absolute()
        print("✅ Follow rate heatmap (absolute positions)")
        
        self.plot_follow_rate_comparison_clean()
        print("✅ Follow rate comparison (clean, sampled)")
        
        # Original plots
        self.plot_follow_rate_by_pattern_and_position()
        print("✅ Follow rate by position & pattern (percentile + heatmap)")
        self.plot_pattern_performance_overview()
        print("✅ Pattern performance overview")
        self.plot_rule_position_vs_text_position()
        print("✅ Rule position vs text position")
        self.plot_follow_rate_by_component_type()
        print("✅ Follow rate by component type")
        self.plot_primacy_recency_bias()
        print("✅ Primacy & recency bias")
        print(f"📊 All visualizations saved to {self.output_dir}")