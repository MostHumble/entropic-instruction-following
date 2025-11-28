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
    
    def plot_follow_rate_comparison_clean(self):
        """Clean comparison: One plot PER rule count, patterns compared within"""
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        n_counts = len(rule_counts)
        fig, axes = plt.subplots(1, n_counts, figsize=(7*n_counts, 6), sharey=True)
        if n_counts == 1:
            axes = [axes]
        
        for count_idx, count in enumerate(rule_counts):
            ax = axes[count_idx]
            count_data = self.expanded_df[self.expanded_df['count'] == count].copy()
            
            patterns = sorted(count_data['pattern'].unique())
            
            for pattern in patterns:
                pattern_data = count_data[count_data['pattern'] == pattern].copy()
                
                max_pos = pattern_data['position_in_rule'].max()
                pattern_data['position_pct'] = (pattern_data['position_in_rule'] / max_pos * 100)
                
                position_stats = pattern_data.groupby(['position_pct', 'trial_id'])['found'].mean().reset_index()
                position_agg = position_stats.groupby('position_pct').agg({
                    'found': ['mean', 'sem', 'count']
                }).reset_index()
                
                position_agg.columns = ['position_pct', 'mean', 'sem', 'count']
                
                step = max(1, len(position_agg) // 40)
                plot_data = position_agg.iloc[::step]
                
                color = self.color_map.get(pattern, '#1f77b4')
                n_trials = int(position_agg['count'].iloc[0])
                
                ax.plot(plot_data['position_pct'], plot_data['mean'], 
                       marker='o', linewidth=2, markersize=4, 
                       color=color, alpha=0.8,
                       label=f"{pattern} (n={n_trials})")
            
            ax.set_xlabel("Position in Rule (%)", fontsize=12)
            ax.set_ylabel("Follow Rate", fontsize=12)
            ax.set_title(f"Rule Length: {count} words", fontsize=13, fontweight='bold')
            ax.legend(title="Pattern", loc='best', fontsize=9, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.suptitle("Follow Rate Comparison (Normalized by Rule Length)", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "01c_follow_rate_comparison_clean.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_follow_rate_by_pattern_and_position(self):
        """Combined view with percentiles (backward compatibility)"""
        # WARNING: This mixes different rule counts - use new plots instead!
        rule_counts = sorted(self.expanded_df['count'].unique())
        
        if len(rule_counts) > 1:
            print(f"⚠️  Warning: This plot mixes rule counts {rule_counts}. Use plot_follow_rate_comparison_clean() instead.")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        max_position = self.expanded_df['position_in_rule'].max()
        rule_length = max_position + 1
        
        ax1 = axes[0]
        
        patterns = sorted(self.expanded_df['pattern'].unique())
        
        for pattern in patterns:
            pattern_data = self.expanded_df[self.expanded_df['pattern'] == pattern]
            
            position_stats = pattern_data.groupby(['position_in_rule', 'trial_id'])['found'].mean().reset_index()
            position_agg = position_stats.groupby('position_in_rule').agg(
                mean=('found', 'mean'),
                sem=('found', 'sem'),
                n=('found', 'count')
            ).reset_index()
            
            position_agg['position_pct'] = (position_agg['position_in_rule'] / max_position) * 100
            
            step = max(1, len(position_agg) // 40)
            plot_data = position_agg.iloc[::step]
            
            color = self.color_map.get(pattern, '#1f77b4')
            n_trials = int(plot_data['n'].iloc[0])
            
            ax1.plot(plot_data['position_pct'], plot_data['mean'], 
                    marker='o', label=f"{pattern} (n={n_trials} trials)", 
                    linewidth=2, markersize=4, alpha=0.8, color=color)
            
            ax1.fill_between(
                plot_data['position_pct'],
                plot_data['mean'] - 1.96 * plot_data['sem'],
                plot_data['mean'] + 1.96 * plot_data['sem'],
                alpha=0.2, color=color
            )
        
        ax1.set_xlabel("Position in Rule (%)", fontsize=12)
        ax1.set_ylabel("Follow Rate", fontsize=12)
        ax1.set_title(f"Follow Rate by Pattern (Max Rule Length: {rule_length} words, with 95% CI)", 
                     fontsize=14, fontweight='bold')
        ax1.legend(title="Pattern", loc='best', fontsize=9, ncol=2)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(-0.05, 1.05)
        ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax2 = axes[1]
        
        trial_agg = self.expanded_df.groupby(['pattern', 'position_in_rule', 'trial_id'])['found'].mean().reset_index()
        final_agg = trial_agg.groupby(['pattern', 'position_in_rule'])['found'].mean().reset_index()
        
        final_agg['position_bin'] = pd.cut(
            final_agg['position_in_rule'], 
            bins=10, 
            labels=[f'{i*10}-{(i+1)*10}%' for i in range(10)],
            duplicates='drop'
        )
        
        heatmap_data = final_agg.groupby(['pattern', 'position_bin'], observed=True)['found'].mean().unstack() 
        
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
            total='sum',      
            n='count',         
            mean='mean',
            std='std'
        ).reset_index()
        component_stats = component_stats.sort_values('mean', ascending=False)
        
        # Dynamic color assignment
        component_colors = self._get_component_colors(component_stats['component_type'].tolist())
        colors = [component_colors[ct] for ct in component_stats['component_type']]
        
        ax.barh(component_stats['component_type'], component_stats['mean'], 
                color=colors, alpha=0.7)
        
        ax.set_xlabel("Follow Rate", fontsize=12)
        ax.set_ylabel("Component Type", fontsize=12)
        ax.set_title("Follow Rate by Component Type (Coherent vs Random)", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)
        
        for i, (ct, mean, n) in enumerate(zip(component_stats['component_type'], 
                                               component_stats['mean'],
                                               component_stats['n'])):
            ax.text(mean + 0.02, i, f'{mean:.1%} (n={int(n)})', 
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
            total='sum',      
            n='count',
            mean='mean',
            std='std'
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
        
        pattern_quintile = self.expanded_df.groupby(['pattern', 'position_quintile'], observed=True)['found'].mean().unstack()
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
    
    def plot_rule_position_vs_text_position(self):
        """Scatter plots: Rule position vs position in text"""
        patterns = sorted(self.expanded_df['pattern'].unique())
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
                lambda x: x[0] if x and len(x) > 0 else -1
            )
            pattern_data = pattern_data[pattern_data['first_pos_in_text'] >= 0]
            
            if len(pattern_data) == 0:
                ax.text(0.5, 0.5, f"No valid positions for: {pattern}", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Pattern: {pattern}")
                continue
            
            ax.scatter(
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
        
        for idx in range(n_patterns, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_rule_position_vs_text_position.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_all(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        self.plot_follow_rate_by_position_absolute()
        print("✅ Follow rate by absolute position (separate subplots)")
        
        self.plot_follow_rate_heatmap_absolute()
        print("✅ Follow rate heatmap (absolute positions)")
        
        self.plot_follow_rate_comparison_clean()
        print("✅ Follow rate comparison (clean, sampled)")
        
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