import pandas as pd
import json
import numpy as np
from scipy import stats

def generate_summary_statistics(results_csv: str, output_file: str = "data/results/analysis_summary.txt"):
    """Generate statistical summary with proper trial aggregation"""
    df = pd.read_csv(results_csv)
    
    # Extract trial number from ID (format: type_count_trial_random)
    df['trial_id'] = df['id'].str.extract(r'_(\d+)_\d+$')[0].astype(int)
    df['config_id'] = df['id'].str.extract(r'^(.+)_\d+_\d+$')[0]
    
    # Expand word details
    expanded_rows = []
    for _, row in df.iterrows():
        word_details = json.loads(row['word_details'])
        for wd in word_details:
            expanded_rows.append({
                'pattern': row['pattern'],
                'count': row['count'],
                'trial_id': row['trial_id'],
                'config_id': row['config_id'],
                'position_in_rule': wd['position'],
                'found': wd['found'],
                'occurrences': wd['occurrences']
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # Aggregate by configuration (pattern + count), averaging across trials
    pattern_stats = df.groupby(['pattern', 'count']).agg({
        'score': ['mean', 'std', 'sem', 'count'],  # sem = standard error of mean
        'passed_count': ['mean', 'std'],
        'total_count': 'first'
    }).reset_index()
    
    # Flatten column names
    pattern_stats.columns = ['_'.join(col).strip('_') for col in pattern_stats.columns]
    
    # Calculate 95% confidence intervals
    pattern_stats['score_ci_lower'] = pattern_stats['score_mean'] - 1.96 * pattern_stats['score_sem']
    pattern_stats['score_ci_upper'] = pattern_stats['score_mean'] + 1.96 * pattern_stats['score_sem']
    
    # Word-level aggregation across trials
    word_level_stats = expanded_df.groupby(['pattern', 'count', 'position_in_rule']).agg({
        'found': ['mean', 'std', 'sem', 'count']
    }).reset_index()
    word_level_stats.columns = ['_'.join(col).strip('_') for col in word_level_stats.columns]
    
    summary = f"""
EXPERIMENT RESULTS SUMMARY (Research-Grade Analysis)
{'='*70}

Overall Statistics:
  - Total samples: {len(df)}
  - Unique configurations: {df['config_id'].nunique()}
  - Trials per configuration: {df.groupby('config_id')['trial_id'].nunique().mean():.1f} (avg)
  - Total words analyzed: {len(expanded_df)}
  - Overall follow rate: {expanded_df['found'].mean():.2%} ± {expanded_df['found'].sem():.4f}
  - Mean score: {df['score'].mean():.2%} ± {df['score'].sem():.4f}

By Pattern (with 95% Confidence Intervals):
"""
    
    for pattern in sorted(pattern_stats['pattern'].unique()):
        pattern_data = pattern_stats[pattern_stats['pattern'] == pattern]
        
        for _, row in pattern_data.iterrows():
            n_trials = int(row['score_count'])
            mean_score = row['score_mean']
            ci_lower = row['score_ci_lower']
            ci_upper = row['score_ci_upper']
            
            summary += f"\n  {pattern:15s} (n={row['count']:.0f} words, trials={n_trials}):"
            summary += f"\n    Score: {mean_score:.2%} [95% CI: {ci_lower:.2%} - {ci_upper:.2%}]"
            summary += f"\n    Std Dev: {row['score_std']:.4f}"
    
    # Statistical tests
    summary += f"\n\n{'='*70}\nStatistical Tests:\n{'='*70}\n"
    
    # ANOVA: Does pattern type affect performance?
    patterns = df['pattern'].unique()
    if len(patterns) > 2:
        pattern_groups = [df[df['pattern'] == p]['score'].values for p in patterns]
        f_stat, p_value = stats.f_oneway(*pattern_groups)
        summary += f"\nOne-way ANOVA (pattern effect):"
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
    
    # Pairwise comparisons (if coherent vs random exist)
    if 'c' in patterns and 'r' in patterns:
        coherent_scores = df[df['pattern'] == 'c']['score'].values
        random_scores = df[df['pattern'] == 'r']['score'].values
        
        t_stat, p_value = stats.ttest_ind(coherent_scores, random_scores)
        cohen_d = (coherent_scores.mean() - random_scores.mean()) / np.sqrt(
            (coherent_scores.std()**2 + random_scores.std()**2) / 2
        )
        
        summary += f"\n\nCoherent vs Random (t-test):"
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
        summary += f"\n  Coherent mean: {coherent_scores.mean():.2%} ± {stats.sem(coherent_scores):.4f}"
        summary += f"\n  Random mean: {random_scores.mean():.2%} ± {stats.sem(random_scores):.4f}"
    
    # Inter-trial reliability (ICC)
    summary += f"\n\n{'='*70}\nReliability Analysis:\n{'='*70}\n"
    
    # Calculate ICC for each pattern
    for pattern in patterns:
        pattern_df = df[df['pattern'] == pattern]
        if pattern_df['trial_id'].nunique() > 1:
            # Reshape for ICC calculation
            pivot_data = pattern_df.pivot_table(
                values='score', 
                index='config_id', 
                columns='trial_id'
            )
            
            if len(pivot_data) > 1:
                # Calculate ICC(2,1) - two-way random effects
                n_configs = len(pivot_data)
                n_trials = pivot_data.shape[1]
                
                grand_mean = pivot_data.values.mean()
                ss_rows = n_trials * ((pivot_data.mean(axis=1) - grand_mean)**2).sum()
                ss_cols = n_configs * ((pivot_data.mean(axis=0) - grand_mean)**2).sum()
                ss_error = ((pivot_data.values - pivot_data.mean(axis=1).values.reshape(-1, 1) - 
                            pivot_data.mean(axis=0).values + grand_mean)**2).sum()
                
                ms_rows = ss_rows / (n_configs - 1)
                ms_error = ss_error / ((n_configs - 1) * (n_trials - 1))
                
                icc = (ms_rows - ms_error) / (ms_rows + (n_trials - 1) * ms_error)
                
                summary += f"\nPattern '{pattern}':"
                summary += f"\n  ICC(2,1): {icc:.4f}"
                if icc > 0.75:
                    summary += " (excellent reliability)"
                elif icc > 0.60:
                    summary += " (good reliability)"
                elif icc > 0.40:
                    summary += " (fair reliability)"
                else:
                    summary += " (poor reliability)"
    
    summary += f"\n\n{'='*70}\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001\n{'='*70}\n"
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"📝 Summary saved to {output_file}")
    return summary