import pandas as pd
import json
import numpy as np
from scipy import stats

def generate_summary_statistics(results_csv: str, output_file: str = "data/results/analysis_summary.txt"):
    """Generate statistical summary with proper hierarchical (seed → trial) aggregation"""
    df = pd.read_csv(results_csv)
    
    # Extract seed and trial from new data structure
    df['seed'] = df.get('seed', 'unknown')
    df['trial_id'] = df.get('trial', 0)
    df['config_id'] = df['seed'].astype(str) + '_' + df['pattern'] + '_' + df['count'].astype(str)
    
    # Expand word details
    expanded_rows = []
    for _, row in df.iterrows():
        word_details = json.loads(row['word_details'])
        for wd in word_details:
            expanded_rows.append({
                'pattern': row['pattern'],
                'count': row['count'],
                'seed': row['seed'],
                'trial_id': row['trial_id'],
                'config_id': row['config_id'],
                'position_in_rule': wd['position'],
                'found': wd['found'],
                'occurrences': wd['occurrences']
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Hierarchical aggregation: pattern + count + seed, averaging across trials within each seed
    seed_level_stats = df.groupby(['pattern', 'count', 'seed']).agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    seed_level_stats.columns = ['_'.join(col).strip('_') for col in seed_level_stats.columns]
    
    # Then aggregate across seeds (pattern + count level)
    pattern_stats = seed_level_stats.groupby(['pattern', 'count']).agg({
        'score_mean': ['mean', 'std', 'sem', 'count'],  # mean of seed means
        'score_std': 'mean'  # average within-seed std
    }).reset_index()
    
    # Flatten column names
    pattern_stats.columns = ['_'.join(col).strip('_') for col in pattern_stats.columns]
    pattern_stats.columns = ['pattern', 'count', 'score_mean', 'score_between_seed_std', 
                             'score_sem', 'n_seeds', 'score_within_seed_std']
    
    # Calculate 95% confidence intervals
    pattern_stats['score_ci_lower'] = pattern_stats['score_mean'] - 1.96 * pattern_stats['score_sem']
    pattern_stats['score_ci_upper'] = pattern_stats['score_mean'] + 1.96 * pattern_stats['score_sem']
    
    # Word-level aggregation: first across trials, then across seeds
    word_level_stats = expanded_df.groupby(['pattern', 'count', 'seed', 'position_in_rule']).agg({
        'found': 'mean'  # mean across trials within seed
    }).reset_index()
    
    word_level_stats_final = word_level_stats.groupby(['pattern', 'count', 'position_in_rule']).agg({
        'found': ['mean', 'std', 'sem', 'count']  # mean across seeds
    }).reset_index()
    word_level_stats_final.columns = ['_'.join(col).strip('_') for col in word_level_stats_final.columns]
    
    summary = f"""
EXPERIMENT RESULTS SUMMARY 
{'='*70}

Overall Statistics:
  - Total samples: {len(df)}
  - Unique seeds: {df['seed'].nunique()}
  - Unique configurations (pattern × count): {df.groupby(['pattern', 'count']).ngroups}
  - Trials per seed per config: {df.groupby(['pattern', 'count', 'seed'])['trial_id'].nunique().mean():.1f} (avg)
  - Total words analyzed: {len(expanded_df)}
  - Overall follow rate: {expanded_df['found'].mean():.2%} ± {expanded_df['found'].sem():.4f}
  - Mean score: {df['score'].mean():.2%} ± {df['score'].sem():.4f}

Data Structure: Seeds → Trials → Patterns → Rule Counts
  - Between-seed variance: Measures consistency across different semantic domains
  - Within-seed variance: Measures consistency across trials from same domain

By Pattern (with 95% Confidence Intervals):
"""
    
    for pattern in sorted(pattern_stats['pattern'].unique()):
        pattern_data = pattern_stats[pattern_stats['pattern'] == pattern]
        
        for _, row in pattern_data.iterrows():
            n_seeds = int(row['n_seeds'])
            mean_score = row['score_mean']
            ci_lower = row['score_ci_lower']
            ci_upper = row['score_ci_upper']
            
            summary += f"\n  {pattern:15s} (n={row['count']:.0f} words, seeds={n_seeds}):"
            summary += f"\n    Score: {mean_score:.2%} [95% CI: {ci_lower:.2%} - {ci_upper:.2%}]"
            summary += f"\n    Between-seed Std: {row['score_between_seed_std']:.4f}"
            summary += f"\n    Within-seed Std: {row['score_within_seed_std']:.4f}"
    
    # Variance decomposition
    summary += f"\n\n{'='*70}\nVariance Decomposition:\n{'='*70}\n"
    
    for pattern in sorted(df['pattern'].unique()):
        pattern_df = df[df['pattern'] == pattern]
        
        # Calculate variance components
        grand_mean = pattern_df['score'].mean()
        
        # Between-seed variance
        seed_means = pattern_df.groupby('seed')['score'].mean()
        var_between_seeds = ((seed_means - grand_mean)**2).sum() / (len(seed_means) - 1)
        
        # Within-seed variance (average of within-seed variances)
        within_seed_vars = pattern_df.groupby('seed')['score'].var()
        var_within_seeds = within_seed_vars.mean()
        
        # Total variance
        var_total = pattern_df['score'].var()
        
        # Intraclass correlation (ICC) - proportion of variance due to seeds
        icc_seeds = var_between_seeds / (var_between_seeds + var_within_seeds) if (var_between_seeds + var_within_seeds) > 0 else 0
        
        summary += f"\nPattern '{pattern}':"
        summary += f"\n  Total variance: {var_total:.6f}"
        summary += f"\n  Between-seed variance: {var_between_seeds:.6f} ({100*var_between_seeds/var_total:.1f}% of total)"
        summary += f"\n  Within-seed variance: {var_within_seeds:.6f} ({100*var_within_seeds/var_total:.1f}% of total)"
        summary += f"\n  ICC (seed consistency): {icc_seeds:.4f}"
        if icc_seeds > 0.75:
            summary += " (high consistency across seeds)"
        elif icc_seeds > 0.5:
            summary += " (moderate consistency across seeds)"
        else:
            summary += " (low consistency - high seed-specific effects)"
    
    # Statistical tests
    summary += f"\n\n{'='*70}\nStatistical Tests:\n{'='*70}\n"
    
    # ANOVA: Does pattern type affect performance? (using seed-level means)
    patterns = seed_level_stats['pattern'].unique()
    if len(patterns) > 2:
        pattern_groups = [seed_level_stats[seed_level_stats['pattern'] == p]['score_mean'].values for p in patterns]
        f_stat, p_value = stats.f_oneway(*pattern_groups)
        summary += f"\nOne-way ANOVA (pattern effect, using seed-level means):"
        summary += f"\n  F-statistic: {f_stat:.4f}"
        summary += f"\n  p-value: {p_value:.4e}"
        if p_value < 0.001:
            summary += " ***"
        elif p_value < 0.01:
            summary += " **"
        elif p_value < 0.05:
            summary += " *"
        summary += "\n  Interpretation: "
        if p_value < 0.05:
            summary += "Pattern type SIGNIFICANTLY affects performance"
        else:
            summary += "No significant effect of pattern type"
        summary += "\n  Note: This test uses seed-level means as independent observations"
    
    # Pairwise comparisons (if coherent vs random exist) - using seed-level means
    if 'c' in patterns and 'r' in patterns:
        # Use seed-level means as independent observations
        coherent_seed_means = seed_level_stats[seed_level_stats['pattern'] == 'c']['score_mean'].values
        random_seed_means = seed_level_stats[seed_level_stats['pattern'] == 'r']['score_mean'].values
        
        # Also get trial-level data for comparison
        coherent_all = df[df['pattern'] == 'c']['score'].values
        random_all = df[df['pattern'] == 'r']['score'].values
        
        t_stat, p_value = stats.ttest_ind(coherent_seed_means, random_seed_means)
        cohen_d = (coherent_seed_means.mean() - random_seed_means.mean()) / np.sqrt(
            (coherent_seed_means.std()**2 + random_seed_means.std()**2) / 2
        )
        
        summary += f"\n\nCoherent vs Random (t-test using seed-level means):"
        summary += f"\n  t-statistic: {t_stat:.4f}"
        summary += f"\n  p-value: {p_value:.4e}"
        if p_value < 0.001:
            summary += " ***"
        elif p_value < 0.01:
            summary += " **"
        elif p_value < 0.05:
            summary += " *"
        summary += f"\n  Cohen's d: {cohen_d:.4f} "
        if abs(cohen_d) > 0.8:
            summary += "(large effect)"
        elif abs(cohen_d) > 0.5:
            summary += "(medium effect)"
        elif abs(cohen_d) > 0.2:
            summary += "(small effect)"
        else:
            summary += "(negligible effect)"
        summary += f"\n  Coherent mean (seed-level): {coherent_seed_means.mean():.2%} ± {stats.sem(coherent_seed_means):.4f}"
        summary += f"\n  Random mean (seed-level): {random_seed_means.mean():.2%} ± {stats.sem(random_seed_means):.4f}"
        summary += f"\n  Coherent mean (all trials): {coherent_all.mean():.2%} ± {stats.sem(coherent_all):.4f}"
        summary += f"\n  Random mean (all trials): {random_all.mean():.2%} ± {stats.sem(random_all):.4f}"
        summary += f"\n  Note: Seed-level analysis treats each seed as independent (n={len(coherent_seed_means)} vs {len(random_seed_means)})"
    
    # Inter-trial reliability within seeds (ICC)
    summary += f"\n\n{'='*70}\nReliability Analysis (Within-Seed Trial Consistency):\n{'='*70}\n"
    
    # Calculate ICC for each pattern, considering seed structure
    for pattern in patterns:
        pattern_df = df[df['pattern'] == pattern]
        
        # Group by seed and calculate within-seed ICC
        seed_iccs = []
        for seed in pattern_df['seed'].unique():
            seed_data = pattern_df[pattern_df['seed'] == seed]
            if seed_data['trial_id'].nunique() > 1 and len(seed_data) > 1:
                # Calculate ICC for this seed
                trials = seed_data['score'].values
                n_trials = len(trials)
                if n_trials > 1:
                    mean_score = trials.mean()
                    var_total = trials.var()
                    # Simple ICC estimate for repeated measures
                    icc_estimate = max(0, 1 - var_total / (mean_score * (1 - mean_score) + 0.001))
                    seed_iccs.append(icc_estimate)
        
        if seed_iccs:
            avg_icc = np.mean(seed_iccs)
            summary += f"\nPattern '{pattern}':"
            summary += f"\n  Average within-seed ICC: {avg_icc:.4f} (across {len(seed_iccs)} seeds)"
            if avg_icc > 0.75:
                summary += " (excellent trial consistency)"
            elif avg_icc > 0.60:
                summary += " (good trial consistency)"
            elif avg_icc > 0.40:
                summary += " (fair trial consistency)"
            else:
                summary += " (poor trial consistency)"
            summary += f"\n  ICC range: [{min(seed_iccs):.4f}, {max(seed_iccs):.4f}]"
    
    summary += f"\n\n{'='*70}\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001\n{'='*70}\n"
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"📝 Summary saved to {output_file}")
    return summary