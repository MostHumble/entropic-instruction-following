import pandas as pd
import json
import numpy as np
from typing import Dict

def analyze_positions(results_csv: str) -> Dict:
    """
    Analyze position-based patterns in rule following with seed awareness.
    
    Returns:
        Dictionary containing various position analysis metrics
    """
    df = pd.read_csv(results_csv)
    
    # Expand word details
    expanded_rows = []
    for _, row in df.iterrows():
        seed = row.get('seed', 'unknown')
        trial_id = row.get('trial', 0)
        
        word_details = json.loads(row['word_details'])
        for wd in word_details:
            expanded_rows.append({
                'pattern': row['pattern'],
                'count': row['count'],
                'seed': seed,
                'trial_id': trial_id,
                'position_in_rule': wd['position'],
                'word': wd['word'],
                'found': wd['found'],
                'positions_in_text': wd['positions_in_text'],
                'occurrences': wd['occurrences'],
                'sample_id': row['id']
            })
    
    expanded_df = pd.DataFrame(expanded_rows)
    
    # 1. Primacy/Recency analysis
    expanded_df['position_quintile'] = pd.qcut(
        expanded_df['position_in_rule'], 
        q=5, 
        duplicates='drop', 
        labels=['1st', '2nd', '3rd', '4th', '5th']
    )
    
    primacy_recency = expanded_df.groupby('position_quintile')['found'].agg(['sum', 'count']).reset_index()
    primacy_recency['follow_rate'] = primacy_recency['sum'] / primacy_recency['count']
    
    # 2. Position ordering analysis (rule position vs text position)
    ordering_data = []
    for pattern in expanded_df['pattern'].unique():
        pattern_data = expanded_df[
            (expanded_df['pattern'] == pattern) & 
            (expanded_df['found'] == True) &
            (expanded_df['occurrences'] > 0)
        ].copy()
        
        if len(pattern_data) > 0:
            pattern_data['first_pos_in_text'] = pattern_data['positions_in_text'].apply(lambda x: x[0] if x else -1)
            pattern_data = pattern_data[pattern_data['first_pos_in_text'] >= 0]
            
            if len(pattern_data) > 2:
                correlation = pattern_data['position_in_rule'].corr(pattern_data['first_pos_in_text'])
                z = np.polyfit(pattern_data['position_in_rule'], pattern_data['first_pos_in_text'], 1)
                
                ordering_data.append({
                    'pattern': pattern,
                    'correlation': correlation,
                    'slope': z[0],
                    'intercept': z[1],
                    'samples': len(pattern_data)
                })
    
    ordering_df = pd.DataFrame(ordering_data)
    
    # 3. By-position analysis across all patterns
    position_stats = expanded_df.groupby('position_in_rule').agg({
        'found': ['sum', 'count', 'mean']
    }).reset_index()
    position_stats.columns = ['position_in_rule', 'followed', 'total', 'follow_rate']
    
    return {
        'expanded_df': expanded_df,
        'primacy_recency': primacy_recency,
        'ordering': ordering_df,
        'position_stats': position_stats
    }


def print_position_analysis_summary(analysis_results: Dict) -> str:
    """Generate a text summary of position analysis"""
    primacy_recency = analysis_results['primacy_recency']
    ordering = analysis_results['ordering']
    
    summary = "\n" + "="*70 + "\nPOSITION ANALYSIS\n" + "="*70 + "\n"
    
    # Primacy/Recency
    summary += "\nPrimacy & Recency Effect:\n"
    for _, row in primacy_recency.iterrows():
        summary += f"  {row['position_quintile']:5s} quintile: {row['follow_rate']:6.2%} follow rate\n"
    
    # Check for primacy bias
    first_quintile = primacy_recency.iloc[0]['follow_rate']
    last_quintile = primacy_recency.iloc[-1]['follow_rate']
    if first_quintile > last_quintile + 0.1:
        summary += f"\n  ⚠️  PRIMACY BIAS DETECTED: First {first_quintile:.2%} vs Last {last_quintile:.2%}\n"
    elif last_quintile > first_quintile + 0.1:
        summary += f"\n  ⚠️  RECENCY BIAS DETECTED: Last {last_quintile:.2%} vs First {first_quintile:.2%}\n"
    else:
        summary += f"\n  ✅ No strong primacy/recency bias detected\n"
    
    # Ordering analysis
    summary += "\nWord Ordering Analysis (Rule Position vs Text Position):\n"
    for _, row in ordering.iterrows():
        summary += f"  {row['pattern']:15s}: correlation={row['correlation']:+.3f}, slope={row['slope']:+.2f}\n"
        if row['correlation'] > 0.3:
            summary += f"                   → Words appear in order (positive correlation)\n"
        elif row['correlation'] < -0.3:
            summary += f"                   → Words appear in reverse order (negative correlation)\n"
        else:
            summary += f"                   → Order is scrambled (weak correlation)\n"
    
    return summary