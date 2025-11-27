import pandas as pd
import json

def generate_summary_statistics(results_csv: str, output_file: str = "data/results/analysis_summary.txt"):
    """Generate statistical summary of results"""
    df = pd.read_csv(results_csv)
    
    # Expand word details
    expanded_rows = []
    for _, row in df.iterrows():
        word_details = json.loads(row['word_details'])
        for wd in word_details:
            expanded_rows.append({
                'pattern': row['pattern'],
                'position_in_rule': wd['position'],
                'found': wd['found'],
                'occurrences': wd['occurrences'],
                'component_type': 'coherent' if wd['word'] in row.get('coherent_words', []) else 'random'
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    summary = f"""
EXPERIMENT RESULTS SUMMARY
{'='*70}

Overall Statistics:
  - Total samples: {len(df)}
  - Total words analyzed: {len(expanded_df)}
  - Overall follow rate: {expanded_df['found'].mean():.2%}
  - Mean score: {df['score'].mean():.2%}
  - Std dev score: {df['score'].std():.4f}

By Pattern:
"""
    
    for pattern in sorted(df['pattern'].unique()):
        pattern_data = df[df['pattern'] == pattern]
        follow_rate = expanded_df[expanded_df['pattern'] == pattern]['found'].mean()
        mean_score = pattern_data['score'].mean()
        summary += f"\n  {pattern:15s}: {follow_rate:6.2%} follow rate, {mean_score:6.2%} mean score (n={len(pattern_data)})"
    
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"📝 Summary saved to {output_file}")
    return summary