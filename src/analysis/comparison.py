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
            if 'model' not in df.columns:
                parts = csv_file.stem.split('_')
                model_name = '_'.join(parts[2:]) if len(parts) >= 3 else 'unknown'
                df['model'] = model_name
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        print(f"📊 Loaded {len(csv_files)} result files, {len(combined)} total samples")
        print(f"   Models: {sorted(combined['model'].unique())}")

        return combined

    def _expand_word_details(self):
        import json
        expanded_rows = []

        for _, row in self.all_results.iterrows():
            trial_id = int(row['id'].split('_')[-2])
            config_id = '_'.join(row['id'].split('_')[:-2])

            word_details = json.loads(row['word_details'])
            for wd in word_details:
                expanded_rows.append({
                    'model': row['model'],
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
        models = sorted(self.all_results['model'].unique())
        if len(models) <= 1:
            print("⚠️  Only one model found, skipping comparison")
            return

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_model_pattern_comparison(ax1)

        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_model_by_rule_length(ax2)

        ax3 = fig.add_subplot(gs[1, :])
        self._plot_model_position_comparison(ax3)

        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_model_primacy_recency(ax4)

        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_model_coherent_vs_random(ax5)

        ax6 = fig.add_subplot(gs[2, 2])
        self._plot_model_summary_table(ax6)

        plt.suptitle("Model Comparison: Comprehensive Analysis",
                     fontsize=18, fontweight='bold', y=0.995)
        plt.savefig(self.results_dir / "model_comparison_comprehensive.png",
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Comprehensive model comparison saved")

    # -------------------------
    # FIXED AGG CALLS BELOW
    # -------------------------

    def _plot_model_pattern_comparison(self, ax):
        model_pattern_scores = (
            self.expanded_df
                .groupby(['model', 'pattern'])['found']
                .agg(mean='mean', sem='sem')
                .reset_index()
        )

        models = sorted(self.all_results['model'].unique())
        patterns = sorted(self.expanded_df['pattern'].unique())

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

    def _plot_model_by_rule_length(self, ax):
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

        ax.set_xlabel("Rule Length (words)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
        ax.set_title("Scaling with Rule Length", fontsize=12, fontweight='bold')
        ax.legend(title="Model", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

    def _plot_model_position_comparison(self, ax):
        models = sorted(self.expanded_df['model'].unique())

        for model in models:
            model_data = self.expanded_df[self.expanded_df['model'] == model].copy()

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

    def _plot_model_coherent_vs_random(self, ax):
        def classify_pattern(pattern):
            if pattern == 'c' or (pattern.startswith('c') and 'r' not in pattern):
                return 'Coherent'
            elif pattern == 'r' or (pattern.startswith('r') and 'c' not in pattern):
                return 'Random'
            else:
                return 'Mixed'

        self.expanded_df['pattern_type'] = self.expanded_df['pattern'].apply(classify_pattern)

        model_type_scores = (
            self.expanded_df
                .groupby(['model', 'pattern_type'])['found']
                .agg(mean='mean', sem='sem')
                .reset_index()
        )

        models = sorted(model_type_scores['model'].unique())
        pattern_types = ['Coherent', 'Random', 'Mixed']

        x = np.arange(len(models))
        width = 0.25

        colors = {'Coherent': '#2ca02c', 'Random': '#d62728', 'Mixed': '#ff7f0e'}

        for idx, ptype in enumerate(pattern_types):
            type_data = model_type_scores[model_type_scores['pattern_type'] == ptype]
            type_data = type_data.set_index('model').reindex(models).reset_index()

            offset = (idx - 1) * width
            ax.bar(
                x + offset, type_data['mean'], width,
                label=ptype, alpha=0.8,
                yerr=1.96 * type_data['sem'], capsize=3,
                color=colors.get(ptype, 'gray')
            )

        ax.set_xlabel("Model", fontsize=11, fontweight='bold')
        ax.set_ylabel("Follow Rate", fontsize=11, fontweight='bold')
        ax.set_title("Coherent vs Random", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend(title="Pattern Type", fontsize=8)
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

    def _plot_model_summary_table(self, ax):
        ax.axis('off')

        summary_stats = []
        for model in sorted(self.all_results['model'].unique()):
            model_expanded = self.expanded_df[self.expanded_df['model'] == model]

            overall_mean = model_expanded['found'].mean()

            pattern_means = model_expanded.groupby('pattern')['found'].mean()
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
