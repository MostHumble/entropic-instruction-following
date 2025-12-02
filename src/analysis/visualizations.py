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
            colors = plt.cm.hsv(np.linspace(0, 0.9, len(patterns)))
        
        self.color_map = {pattern: colors[i] for i, pattern in enumerate(patterns)}
        
        # Override with semantic colors if pattern type is obvious
        semantic_colors = {
            'c': '#2ca02c',      # Green for coherent
            'r': '#d62728',      # Red for random
        }
        
        for pattern in patterns:
            if pattern in semantic_colors:
                self.color_map[pattern] = semantic_colors[pattern]
            elif pattern.startswith('c') and 'r' not in pattern.lower():
                self.color_map[pattern] = plt.cm.Greens(0.5 + 0.3 * (hash(pattern) % 5) / 5)
            elif pattern.startswith('r') and 'c' not in pattern.lower():
                self.color_map[pattern] = plt.cm.Reds(0.5 + 0.3 * (hash(pattern) % 5) / 5)
    
    def _expand_word_details(self):
        """Expand word_details JSON into separate rows for analysis"""
        expanded_rows = []
        
        for _, row in self.df.iterrows():
            # Extract seed and trial from the row (new data structure)
            seed = row.get('seed', 'unknown')
            trial_id = row.get('trial', 0)
            
            # Create config_id without trial info for grouping
            config_id = f"{seed}_{row['pattern']}_{row['count']}"
            
            word_details = json.loads(row['word_details'])
            for wd in word_details:
                expanded_rows.append({
                    'pattern': row['pattern'],
                    'count': row['count'], 
                    'score': row['score'],
                    'seed': seed,
                    'trial_id': trial_id,
                    'position_in_rule': wd['position'],
                    'word': wd['word'],
                    'found': wd['found'],
                    'positions_in_text': wd['positions_in_text'],
                    'occurrences': wd['occurrences'],
                    'sample_id': row['id'],
                    'config_id': config_id
                })
        
        self.expanded_df = pd.DataFrame(expanded_rows)
    
    def plot_follow_rate_by_position_absolute(self):
        """Separate subplots for each pattern, ORGANIZED BY RULE COUNT"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        for count in rule_counts:
            count_data = self.expanded_df[self.expanded_df['count'] == count]
            
            # Create subdirectory for this count
            count_dir = self.output_dir / f"{count}_rules"
            count_dir.mkdir(parents=True, exist_ok=True)
            
            # Get patterns for this count
            patterns = sorted(count_data['pattern'].unique())
            n_patterns = len(patterns)
            
            n_cols = 3
            n_rows = (n_patterns + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), 
                                    sharex=False, sharey=True)
            if n_patterns == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, pattern in enumerate(patterns):
                ax = axes[idx]
                pattern_data = count_data[count_data['pattern'] == pattern]
                
                max_position = pattern_data['position_in_rule'].max()
                
                position_stats = pattern_data.groupby(['position_in_rule', 'trial_id'])['found'].mean().reset_index()
                position_agg = position_stats.groupby('position_in_rule').agg({
                    'found': ['mean', 'sem', 'count']
                }).reset_index()
                
                position_agg.columns = ['position', 'mean', 'sem', 'count']
                
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
                ax.set_title(f"Pattern: {pattern} (n={n_trials} trials)", 
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
            
            for idx in range(n_patterns, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f"Follow Rate by Absolute Position ({count} rules)", 
                        fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(count_dir / "01a_follow_rate_by_absolute_position.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_follow_rate_heatmap_absolute(self):
        """Heatmap with absolute positions, ONE PER RULE COUNT"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        for count in rule_counts:
            count_dir = self.output_dir / f"{count}_rules"
            count_dir.mkdir(parents=True, exist_ok=True)
            
            count_data = self.expanded_df[self.expanded_df['count'] == count].copy()
            
            fig, ax = plt.subplots(figsize=(16, 6))
            
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
            ax.set_title(f"Follow Rate Heatmap ({count} rules)", 
                        fontsize=13, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(count_dir / "01b_follow_rate_heatmap_absolute.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_pattern_performance_overview(self):
        """Bar plot: Overall pattern performance BY RULE COUNT"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        for count in rule_counts:
            count_dir = self.output_dir / f"{count}_rules"
            count_dir.mkdir(parents=True, exist_ok=True)
            
            count_data = self.expanded_df[self.expanded_df['count'] == count]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            pattern_scores = count_data.groupby('pattern')['found'].agg(
                mean='mean',
                std='std',
                n='count'
            ).reset_index()
            
            patterns = sorted(pattern_scores['pattern'].unique())
            x = np.arange(len(patterns))
            
            # Use pattern colors
            colors = [self.color_map.get(p, 'steelblue') for p in patterns]
            
            ax.bar(x, pattern_scores['mean'], 
                   alpha=0.8,
                   yerr=pattern_scores['std'], capsize=3,
                   color=colors)
            
            ax.set_xlabel("Pattern Type", fontsize=12)
            ax.set_ylabel("Follow Rate", fontsize=12)
            ax.set_title(f"Overall Follow Rate by Pattern ({count} rules)", 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(patterns, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(count_dir / "01d_pattern_performance.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_rule_position_vs_text_position(self):
        """Scatter plots: ORGANIZED BY RULE COUNT"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        for count in rule_counts:
            count_dir = self.output_dir / f"{count}_rules"
            count_dir.mkdir(parents=True, exist_ok=True)
            
            count_data = self.expanded_df[self.expanded_df['count'] == count]
            patterns = sorted(count_data['pattern'].unique())
            n_patterns = len(patterns)
            
            n_cols = 3
            n_rows = (n_patterns + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            if n_patterns == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, pattern in enumerate(patterns):
                ax = axes[idx]
                
                pattern_data = count_data[
                    (count_data['pattern'] == pattern) &
                    (count_data['found'] == True) &
                    (count_data['occurrences'] > 0)
                ].copy()
                
                if len(pattern_data) == 0:
                    ax.text(0.5, 0.5, f"No data\nPattern: {pattern}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"{pattern}")
                    continue
                
                pattern_data['first_pos_in_text'] = pattern_data['positions_in_text'].apply(
                    lambda x: x[0] if x and len(x) > 0 else -1
                )
                pattern_data = pattern_data[pattern_data['first_pos_in_text'] >= 0]
                
                if len(pattern_data) == 0:
                    ax.text(0.5, 0.5, f"No valid positions\nPattern: {pattern}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"{pattern}")
                    continue
                
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
                
                if len(pattern_data) > 2:
                    z = np.polyfit(pattern_data['position_in_rule'], 
                                  pattern_data['first_pos_in_text'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(pattern_data['position_in_rule'].min(), 
                                         pattern_data['position_in_rule'].max(), 100)
                    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                           label=f'Trend (slope={z[0]:.2f})')
                    
                    corr = np.corrcoef(pattern_data['position_in_rule'], 
                                      pattern_data['first_pos_in_text'])[0, 1]
                    ax.text(0.05, 0.95, f'r={corr:.3f}', 
                           transform=ax.transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.set_xlabel("Position in Rule", fontsize=10)
                ax.set_ylabel("Character Position in Text", fontsize=10)
                ax.set_title(f"Pattern: {pattern}", fontsize=11, fontweight='bold')
                if len(pattern_data) > 2:
                    ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
            
            for idx in range(n_patterns, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f"Rule Position vs Text Position ({count} rules)", 
                        fontsize=15, fontweight='bold', y=1.00)
            plt.tight_layout()
            plt.savefig(count_dir / "02_rule_position_vs_text_position.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_seed_variance_analysis(self):
        """Visualize variance across seeds vs within seeds - split into separate plots"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        for count in rule_counts:
            count_dir = self.output_dir / f"{count}_rules"
            seed_variance_dir = count_dir / "seed_variance"
            seed_variance_dir.mkdir(parents=True, exist_ok=True)
            
            count_data = self.expanded_df[self.expanded_df['count'] == count]
            patterns = sorted(count_data['pattern'].unique())
            
            # Plot 1: Seed-level means with cleaned seed names
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            seed_means = count_data.groupby(['pattern', 'seed'])['found'].mean().reset_index()
            
            # Get unique seeds and clean the names
            unique_seeds = sorted(seed_means['seed'].unique())
            seed_display_names = [seed.split(".")[0] for seed in unique_seeds]
            seed_to_idx = {seed: idx for idx, seed in enumerate(unique_seeds)}
            
            for pattern in patterns:
                pattern_seeds = seed_means[seed_means['pattern'] == pattern]
                x_pos = [seed_to_idx[seed] for seed in pattern_seeds['seed']]
                color = self.color_map.get(pattern, 'steelblue')
                ax.scatter(x_pos, pattern_seeds['found'], label=pattern, 
                          alpha=0.6, s=100, color=color)
            
            ax.set_xlabel("Seed", fontsize=11, fontweight='bold')
            ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
            ax.set_title(f"Seed-Level Performance Variation ({count} rules)", fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(unique_seeds)))
            ax.set_xticklabels(seed_display_names, rotation=45, ha='right')
            ax.legend(title="Pattern", fontsize=9)
            ax.set_ylim(0, 1.0)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(seed_variance_dir / "01_seed_level_means.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Variance decomposition
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            variance_data = []
            for pattern in patterns:
                pattern_df = count_data[count_data['pattern'] == pattern]
                
                # Between-seed variance
                seed_means_vals = pattern_df.groupby('seed')['found'].mean()
                var_between = seed_means_vals.var()
                
                # Within-seed variance
                within_seed_vars = pattern_df.groupby('seed')['found'].var()
                var_within = within_seed_vars.mean()
                
                variance_data.append({
                    'pattern': pattern,
                    'between_seed': var_between,
                    'within_seed': var_within
                })
            
            var_df = pd.DataFrame(variance_data)
            x = np.arange(len(var_df))
            width = 0.35
            
            ax.bar(x - width/2, var_df['between_seed'], width, 
                  label='Between-seed variance', alpha=0.8)
            ax.bar(x + width/2, var_df['within_seed'], width,
                  label='Within-seed variance', alpha=0.8)
            
            ax.set_xlabel("Pattern", fontsize=11, fontweight='bold')
            ax.set_ylabel("Variance", fontsize=11, fontweight='bold')
            ax.set_title(f"Variance Decomposition ({count} rules)", fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(var_df['pattern'], rotation=45, ha='right')
            ax.legend(fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(seed_variance_dir / "02_variance_decomposition.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 3: ICC (consistency) across patterns
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            icc_data = []
            for pattern in patterns:
                pattern_df = count_data[count_data['pattern'] == pattern]
                seed_means_vals = pattern_df.groupby('seed')['found'].mean()
                var_between = seed_means_vals.var()
                within_seed_vars = pattern_df.groupby('seed')['found'].var()
                var_within = within_seed_vars.mean()
                
                icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
                icc_data.append({'pattern': pattern, 'icc': icc})
            
            icc_df = pd.DataFrame(icc_data)
            colors_list = [self.color_map.get(p, 'steelblue') for p in icc_df['pattern']]
            
            ax.bar(range(len(icc_df)), icc_df['icc'], alpha=0.8, color=colors_list)
            ax.set_xlabel("Pattern", fontsize=11, fontweight='bold')
            ax.set_ylabel("ICC (Seed Consistency)", fontsize=11, fontweight='bold')
            ax.set_title(f"Intraclass Correlation - Seed Consistency ({count} rules)", fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(icc_df)))
            ax.set_xticklabels(icc_df['pattern'], rotation=45, ha='right')
            ax.set_ylim(0, 1.0)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, 
                      label='Moderate consistency')
            ax.axhline(y=0.75, color='green', linestyle='--', alpha=0.5,
                      label='High consistency')
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(seed_variance_dir / "03_icc_consistency.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 4: Box plot showing distribution across seeds
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            box_data = []
            box_labels = []
            for pattern in patterns:
                pattern_seeds = seed_means[seed_means['pattern'] == pattern]['found'].values
                box_data.append(pattern_seeds)
                box_labels.append(pattern)
            
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            for patch, pattern in zip(bp['boxes'], box_labels):
                patch.set_facecolor(self.color_map.get(pattern, 'steelblue'))
                patch.set_alpha(0.6)
            
            ax.set_xlabel("Pattern", fontsize=11, fontweight='bold')
            ax.set_ylabel("Follow Rate (Seed-Level)", fontsize=11, fontweight='bold')
            ax.set_title(f"Distribution of Seed-Level Performance ({count} rules)", fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(seed_variance_dir / "04_seed_distribution_boxplot.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Seed variance analysis saved to {seed_variance_dir}/")

    def create_all(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        self.plot_follow_rate_by_position_absolute()
        print("✅ Follow rate by absolute position (organized by rule count)")
        
        self.plot_follow_rate_heatmap_absolute()
        print("✅ Follow rate heatmap (organized by rule count)")
                        
        self.plot_pattern_performance_overview()
        print("✅ Pattern performance overview (organized by rule count)")
        
        self.plot_rule_position_vs_text_position()
        print("✅ Rule position vs text position (organized by rule count)")
        
        self.plot_seed_variance_analysis()
        print("✅ Seed variance analysis (NEW)")
        
        print(f"📊 All visualizations saved to {self.output_dir}")