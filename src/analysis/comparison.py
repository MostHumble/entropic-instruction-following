import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict

class MultiModelComparison:
    """Compare results across multiple models"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.all_results = self._load_all_results()

        if len(self.all_results) == 0:
            raise FileNotFoundError(f"No results CSV files found in {results_dir}")

        self._expand_word_details()
        self._setup_color_scheme()

    def _load_all_results(self) -> pd.DataFrame:
        csv_files = list(self.results_dir.glob("results_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No results_*.csv files found in {self.results_dir}")

        dfs = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['model'] = df['model_name_full'].apply(lambda x: x.split('/')[-1].lower().replace('instruct','it'))
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        print(f" Loaded {len(csv_files)} result files, {len(combined)} total samples")
        print(f"   Models: {sorted(combined['model'].unique())}")

        return combined

    def _expand_word_details(self):
        import json
        expanded_rows = []

        for _, row in self.all_results.iterrows():
            # Extract seed and trial from the row (new data structure)
            seed = row.get('seed', 'unknown')
            trial_id = row.get('trial', 0)
            
            # Create config_id without trial info for grouping
            config_id = f"{row['model']}_{seed}_{row['pattern']}_{row['count']}"

            word_details = json.loads(row['word_details'])
            for wd in word_details:
                expanded_rows.append({
                    'model': row['model'],
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

    def _setup_color_scheme(self):
        models = sorted(self.all_results['model'].unique())
        if len(models) <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
        elif len(models) <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))
        else:
            colors = plt.cm.hsv(np.linspace(0, 0.9, len(models)))

        self.model_colors = {model: colors[i] for i, model in enumerate(models)}

    def plot_model_comparison_comprehensive(self):
        """Comprehensive model comparison - SEPARATED by rule count, individual plots"""
        models = sorted(self.all_results['model'].unique())
        if len(models) <= 1:
            print("⚠️  Only one model found, skipping comparison")
            return

        counts = sorted(self.expanded_df['count'].unique())
        
        # Create individual plots per rule count
        for count in counts:
            count_data = self.expanded_df[self.expanded_df['count'] == count]
            
            # Create subdirectory for this count
            count_dir = self.results_dir / f"comparison_{count}_rules"
            count_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n📊 Generating comparisons for {count}rules...")
            
            # 1. Pattern comparison (bar chart)
            fig, ax = plt.subplots(figsize=(12, 6))
            self._plot_model_pattern_comparison(ax, count_data)
            plt.tight_layout()
            plt.savefig(count_dir / "01_pattern_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Pattern comparison")
            
            # 2. Pattern heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_model_by_pattern_heatmap(ax, count_data)
            plt.tight_layout()
            plt.savefig(count_dir / "02_pattern_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Pattern heatmap")
            
            # 3. Position-based follow rate
            fig, ax = plt.subplots(figsize=(14, 6))
            self._plot_model_position_comparison(ax, count_data)
            plt.tight_layout()
            plt.savefig(count_dir / "03_position_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Position comparison")
            
            # 4. Primacy/Recency effect
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_model_primacy_recency(ax, count_data)
            plt.tight_layout()
            plt.savefig(count_dir / "04_primacy_recency.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Primacy/recency")
            
            # 5. Coherent vs Random
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_model_coherent_vs_random(ax, count_data)
            plt.tight_layout()
            plt.savefig(count_dir / "05_coherent_vs_random.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Coherent vs random")
            
            # 6. Summary table
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_model_summary_table(ax, count_data)
            plt.tight_layout()
            plt.savefig(count_dir / "06_summary_table.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ Summary table")
            
            print(f"📁 Saved to: {count_dir}")
        
        # Also create cross-count comparison
        print(f"\n📊 Generating cross-count comparisons...")
        self._plot_rule_length_comparison_all()
        self.plot_absolute_rules_followed()  # Add this line
        print(f"✅ All model comparisons complete!")

    def _plot_model_pattern_comparison(self, ax, data):
        """Compare models across patterns for SPECIFIC rule count"""
        models = sorted(data['model'].unique())
        patterns = sorted(data['pattern'].unique())
        
        model_pattern_scores = (
            data
                .groupby(['model', 'pattern'])['found']
                .agg(mean='mean', sem='sem')
                .reset_index()
        )

        x = np.arange(len(patterns))
        width = 0.8 / len(models)

        for idx, model in enumerate(models):
            model_data = model_pattern_scores[model_pattern_scores['model'] == model]
            model_data = model_data.set_index('pattern').reindex(patterns).reset_index()

            offset = (idx - len(models)/2 + 0.5) * width
            color = self.model_colors[model]

            ax.bar(
                x + offset, model_data['mean'], width,
                label=model, alpha=0.8,
                yerr=1.96 * model_data['sem'], capsize=3,
                color=color
            )

        ax.set_xlabel("Pattern Type", fontsize=11, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
        ax.set_title("Performance by Pattern (with 95% CI)", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(patterns, rotation=45, ha='right')
        ax.legend(title="Model", loc='best', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    def _plot_model_by_pattern_heatmap(self, ax, data):
        """Heatmap: models vs patterns for specific count"""
        pivot = data.groupby(['model', 'pattern'])['found'].mean().unstack()
        
        sns.heatmap(
            pivot,
            cmap='RdYlGn',
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Follow Rate'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        ax.set_xlabel("Pattern", fontsize=11, fontweight='bold')
        ax.set_ylabel("Model", fontsize=11, fontweight='bold')
        ax.set_title("Model × Pattern Heatmap", fontsize=12, fontweight='bold')

    def _plot_model_position_comparison(self, ax, data):
        """Position comparison for specific rule count"""
        models = sorted(data['model'].unique())

        for model in models:
            model_data = data[data['model'] == model].copy()

            max_pos = model_data['position_in_rule'].max()
            model_data['position_pct'] = (model_data['position_in_rule'] / max_pos * 100).astype(int)

            position_stats = model_data.groupby(['position_pct', 'trial_id'])['found'].mean().reset_index()

            position_agg = (
                position_stats
                    .groupby('position_pct')['found']
                    .agg(mean='mean', sem='sem')
                    .reset_index()
            )

            step = max(1, len(position_agg) // 20)
            plot_data = position_agg.iloc[::step]

            color = self.model_colors[model]
            ax.plot(
                plot_data['position_pct'], plot_data['mean'],
                marker='o', linewidth=2, markersize=4,
                label=model, color=color, alpha=0.8
            )

        ax.set_xlabel("Position in Rule (%)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
        ax.set_title("Position-Based Follow Rate (Normalized)", fontsize=12, fontweight='bold')
        ax.legend(title="Model", loc='best', fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

    def _plot_model_primacy_recency(self, ax, data):
        """Primacy/recency for specific count"""
        data = data.copy()
        data['position_quintile'] = pd.qcut(
            data['position_in_rule'], 
            q=5, 
            duplicates='drop', 
            labels=['1st', '2nd', '3rd', '4th', '5th']
        )
        
        model_quintile = data.groupby(['model', 'position_quintile'], observed=True)['found'].mean().unstack()
        
        model_quintile.plot(kind='bar', ax=ax, alpha=0.8, width=0.8)
        ax.set_xlabel("Model", fontsize=11, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
        ax.set_title("Primacy/Recency Effect", fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.legend(title='Position', fontsize=8, ncol=2)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    def _plot_model_coherent_vs_random(self, ax, data):
        """Show all patterns individually for specific count"""
        model_pattern_scores = (
            data
                .groupby(['model', 'pattern'])['found']
                .agg(mean='mean', sem='sem')
                .reset_index()
        )

        models = sorted(model_pattern_scores['model'].unique())
        patterns = sorted(model_pattern_scores['pattern'].unique())

        x = np.arange(len(models))
        width = 0.8 / len(patterns)

        # Generate distinct colors for each pattern
        if len(patterns) <= 10:
            colors_array = plt.cm.tab10(np.linspace(0, 1, 10))
        else:
            colors_array = plt.cm.tab20(np.linspace(0, 1, 20))
        
        pattern_colors = {pattern: colors_array[i] for i, pattern in enumerate(patterns)}

        for idx, pattern in enumerate(patterns):
            pattern_data = model_pattern_scores[model_pattern_scores['pattern'] == pattern]
            
            positions = x + (idx - len(patterns)/2 + 0.5) * width
            
            ax.bar(
                positions,
                pattern_data['mean'],
                width,
                label=pattern,
                color=pattern_colors[pattern],
                alpha=0.8,
                yerr=1.96 * pattern_data['sem'],
                capsize=3
            )

        ax.set_xlabel("Model", fontsize=11, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
        ax.set_title("All Patterns by Model", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title="Pattern", fontsize=8, ncol=2)
        ax.set_ylim(0.0, 0.6)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    def _plot_model_summary_table(self, ax, data):
        """Summary table for specific count"""
        ax.axis('off')

        summary_stats = []
        for model in sorted(data['model'].unique()):
            model_data = data[data['model'] == model]

            overall_mean = model_data['found'].mean()

            pattern_means = model_data.groupby('pattern')['found'].mean()
            best_pattern = pattern_means.idxmax()
            best_score = pattern_means.max()

            worst_pattern = pattern_means.idxmin()
            worst_score = pattern_means.min()

            summary_stats.append([
                model[:20],
                f'{overall_mean:.1%}',
                f'{best_pattern}\n({best_score:.1%})',
                f'{worst_pattern}\n({worst_score:.1%})'
            ])

        table = ax.table(
            cellText=summary_stats,
            colLabels=['Model', 'Overall', 'Best Pattern', 'Worst Pattern'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        for i in range(1, len(summary_stats) + 1):
            for j in range(4):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')

        ax.set_title("Summary Statistics", fontsize=12, fontweight='bold', pad=20)

    def _plot_rule_length_comparison_all(self):
        """plots showing how models scale with rule length"""
        
        # Plot 1: Overall scaling (all patterns)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        model_count_scores = (
            self.expanded_df
                .groupby(['model', 'count'])['found']
                .agg(mean='mean', sem='sem')
                .reset_index()
        )

        models = sorted(model_count_scores['model'].unique())

        for model in models:
            model_data = model_count_scores[model_count_scores['model'] == model].sort_values('count')
            color = self.model_colors[model]

            ax.plot(
                model_data['count'], model_data['mean'],
                marker='o', linewidth=2, markersize=8,
                label=model, color=color, alpha=0.8
            )

            ax.fill_between(
                model_data['count'],
                model_data['mean'] - 1.96 * model_data['sem'],
                model_data['mean'] + 1.96 * model_data['sem'],
                alpha=0.2, color=color
            )

        ax.set_xlabel("Rule Length (words)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Overall Follow Rate", fontsize=12, fontweight='bold')
        ax.set_title("Model Scaling with Rule Length (All Patterns)", fontsize=13, fontweight='bold')
        ax.legend(title="Model", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison_overall_scaling.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Overall scaling comparison saved")
        
        # Plot 2: Pattern-specific scaling (coherent vs random)
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        def classify_pattern(pattern):
            if pattern in ['c', 'r']:
                return pattern
            return 'mixed'
        
        scaling_data = self.expanded_df.copy()
        scaling_data['pattern_class'] = scaling_data['pattern'].apply(classify_pattern)
        scaling_data = scaling_data[scaling_data['pattern_class'].isin(['c', 'r'])]
        
        for model in models:
            color = self.model_colors[model]
            
            # Coherent (solid line, circles)
            coherent_data = scaling_data[
                (scaling_data['model'] == model) & 
                (scaling_data['pattern_class'] == 'c')
            ]
            
            if len(coherent_data) > 0:
                count_stats = (
                    coherent_data
                        .groupby('count')['found']
                        .agg(mean='mean', sem='sem')
                        .reset_index()
                        .sort_values('count')
                )
                
                ax.plot(
                    count_stats['count'], count_stats['mean'],
                    marker='o', linewidth=2.5, markersize=8,
                    linestyle='-', label=f"{model} (coherent)", 
                    color=color, alpha=0.8
                )
            
            # Random (dashed line, squares, lighter)
            random_data = scaling_data[
                (scaling_data['model'] == model) & 
                (scaling_data['pattern_class'] == 'r')
            ]
            
            if len(random_data) > 0:
                count_stats = (
                    random_data
                        .groupby('count')['found']
                        .agg(mean='mean', sem='sem')
                        .reset_index()
                        .sort_values('count')
                )
                
                ax.plot(
                    count_stats['count'], count_stats['mean'],
                    marker='s', linewidth=2.5, markersize=7,
                    linestyle='--', label=f"{model} (random)", 
                    color=color, alpha=0.5
                )
        
        ax.set_xlabel("Rule Length (words)", fontsize=12, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=12, fontweight='bold')
        ax.set_title("Model Scaling: Coherent (○, solid) vs Random (□, dashed)", fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, ncol=1, loc='best')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison_coherent_vs_random_scaling.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Coherent vs random scaling comparison saved")

    def get_summary(self) -> str:
        summary = ["=" * 60, "MODEL COMPARISON SUMMARY", "=" * 60, ""]

        for model in sorted(self.all_results['model'].unique()):
            model_data = self.all_results[self.all_results['model'] == model]
            model_expanded = self.expanded_df[self.expanded_df['model'] == model]

            mean_score = model_data['score'].mean()
            std_score = model_data['score'].std()
            n_samples = len(model_data)

            summary.append(f"{model}:")
            summary.append(f"  Mean score: {mean_score:.2%} (±{std_score:.4f})")
            summary.append(f"  Samples: {n_samples}")
            summary.append("")

        return "\n".join(summary)

    def plot_model_comparison(self):
        self.plot_model_comparison_comprehensive()

    def plot_absolute_rules_followed(self):
        """Plot absolute number of rules followed by each model across rule counts"""
        
        # Calculate absolute number of rules followed per sample
        self.all_results['absolute_rules_followed'] = self.all_results['score'] * self.all_results['count']
        
        # Plot 1: Overall (all patterns combined)
        output_path = self.results_dir / "model_comparison_absolute_rules_followed.png"
        
        absolute_stats = (
            self.all_results
                .groupby(['model', 'count'])
                .agg({
                    'absolute_rules_followed': ['mean', 'std', 'sem'],
                    'score': 'count'
                })
                .reset_index()
        )
        
        absolute_stats.columns = ['model', 'count', 'mean_rules', 'std_rules', 'sem_rules', 'n_samples']
        
        models = sorted(absolute_stats['model'].unique())
        counts = sorted(absolute_stats['count'].unique())
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        x = np.arange(len(counts))
        width = 0.8 / len(models)
        
        for idx, model in enumerate(models):
            model_data = absolute_stats[absolute_stats['model'] == model].sort_values('count')
            model_data = model_data.set_index('count').reindex(counts).reset_index()
            
            offset = (idx - len(models)/2 + 0.5) * width
            color = self.model_colors[model]
            
            ax.bar(
                x + offset, 
                model_data['mean_rules'], 
                width,
                label=model,
                alpha=0.8,
                color=color,
                yerr=1.96 * model_data['sem_rules'],
                capsize=3
            )
        
        ax.set_xlabel("Number of Rules in Prompt", fontsize=13, fontweight='bold')
        ax.set_ylabel("Absolute Number of Rules Followed", fontsize=13, fontweight='bold')
        ax.set_title("Absolute Rules Followed by Model and Rule Count (All Patterns)", fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(counts)
        ax.legend(title="Model", fontsize=10, ncol=2)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Absolute rules followed comparison saved to {output_path}")
        
        # Summary table for overall
        summary_table = absolute_stats.pivot_table(
            index='model',
            columns='count',
            values='mean_rules',
            aggfunc='first'
        )
        
        summary_path = self.results_dir / "absolute_rules_followed_summary.csv"
        summary_table.to_csv(summary_path)
        print(f"✅ Summary table saved to {summary_path}")
        
        # Plot 2: By pattern - line plots showing scaling
        output_path_pattern = self.results_dir / "model_comparison_absolute_rules_by_pattern.png"
    
        pattern_stats = (
            self.all_results
                .groupby(['model', 'pattern', 'count'])
                .agg({
                    'absolute_rules_followed': ['mean', 'sem'],
                    'score': 'count'
                })
                .reset_index()
        )
    
        pattern_stats.columns = ['model', 'pattern', 'count', 'mean_rules', 'sem_rules', 'n_samples']
    
        patterns = sorted(pattern_stats['pattern'].unique())
        n_patterns = len(patterns)
    
        # Create subplots: one for each pattern
        n_cols = min(3, n_patterns)
        n_rows = (n_patterns + n_cols - 1) // n_cols
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows), squeeze=False)
        axes = axes.flatten()
    
        for idx, pattern in enumerate(patterns):
            ax = axes[idx]
            pattern_data = pattern_stats[pattern_stats['pattern'] == pattern]
        
            for model in models:
                model_data = pattern_data[pattern_data['model'] == model].sort_values('count')
            
                if len(model_data) > 0:
                    color = self.model_colors[model]
                
                    ax.plot(
                        model_data['count'], 
                        model_data['mean_rules'],
                        marker='o',
                        linewidth=2.5,
                        markersize=8,
                        label=model,
                        color=color,
                        alpha=0.8
                    )
                
                    # Add error bars
                    ax.fill_between(
                        model_data['count'],
                        model_data['mean_rules'] - 1.96 * model_data['sem_rules'],
                        model_data['mean_rules'] + 1.96 * model_data['sem_rules'],
                        alpha=0.2,
                        color=color
                    )
        
            ax.set_xlabel("Number of Rules", fontsize=11, fontweight='bold')
            ax.set_ylabel("Absolute Rules Followed", fontsize=11, fontweight='bold')
            ax.set_title(f"Pattern: {pattern}", fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
    
        # Hide extra subplots
        for idx in range(n_patterns, len(axes)):
            axes[idx].set_visible(False)
    
        plt.suptitle("Absolute Rules Followed by Pattern", fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path_pattern, dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"✅ Pattern-specific absolute rules comparison saved to {output_path_pattern}")
    
        # Plot 3: Heatmap showing model x pattern x count
        for count in counts:
            count_data = pattern_stats[pattern_stats['count'] == count]
        
            heatmap_data = count_data.pivot_table(
                index='model',
                columns='pattern',
                values='mean_rules',
                aggfunc='first'
            )
        
            fig, ax = plt.subplots(figsize=(10, 6))
        
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt='.1f',
                cmap='YlOrRd',
                cbar_kws={'label': 'Absolute Rules Followed'},
                ax=ax,
                linewidths=0.5
            )
        
            ax.set_xlabel("Pattern Type", fontsize=12, fontweight='bold')
            ax.set_ylabel("Model", fontsize=12, fontweight='bold')
            ax.set_title(f"Absolute Rules Followed: Model × Pattern ({count} rules)", 
                        fontsize=13, fontweight='bold')
        
            plt.tight_layout()
            output_heatmap = self.results_dir / f"absolute_rules_heatmap_{count}_rules.png"
            plt.savefig(output_heatmap, dpi=300, bbox_inches='tight')
            plt.close()
        
            print(f"✅ Heatmap for {count} rules saved to {output_heatmap}")
    
        # Detailed CSV with pattern breakdowns
        detailed_summary = pattern_stats.pivot_table(
            index=['model', 'pattern'],
            columns='count',
            values='mean_rules',
            aggfunc='first'
        )
    
        detailed_path = self.results_dir / "absolute_rules_followed_by_pattern_detailed.csv"
        detailed_summary.to_csv(detailed_path)
        print(f"✅ Detailed pattern summary saved to {detailed_path}")
    
        return summary_table
