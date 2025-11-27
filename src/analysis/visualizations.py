import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
import os
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
                    'sample_id': row['id']
                })
        
        self.expanded_df = pd.DataFrame(expanded_rows)
    
    def plot_follow_rate_by_position_pattern(self):
        """Heatmap: Follow rate by position in rule and pattern type"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        follow_by_pos_pattern = self.expanded_df.groupby(['pattern', 'position_in_rule'])['found'].agg(['sum', 'count']).reset_index()
        follow_by_pos_pattern['follow_rate'] = follow_by_pos_pattern['sum'] / follow_by_pos_pattern['count']
        
        heatmap_data = follow_by_pos_pattern.pivot(index='pattern', columns='position_in_rule', values='follow_rate')
        
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', cbar_kws={'label': 'Follow Rate'}, ax=ax)
        ax.set_title("Rule Following Rate by Position in Rule and Pattern Type")
        ax.set_xlabel("Position in Rule (0 = First Word)")
        ax.set_ylabel("Pattern Type")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_follow_rate_by_position_pattern.png", dpi=300)
        plt.close()
    
    def plot_rule_position_vs_text_position(self):
        """Scatter plots: Rule position vs position in text"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, pattern in enumerate(self.expanded_df['pattern'].unique()[:4]):
            ax = axes[idx]
            pattern_data = self.expanded_df[
                (self.expanded_df['pattern'] == pattern) & 
                (self.expanded_df['found'] == True) &
                (self.expanded_df['occurrences'] > 0)
            ].copy()
            
            if len(pattern_data) == 0:
                ax.text(0.5, 0.5, f"No data for pattern: {pattern}", ha='center', va='center')
                ax.set_title(f"Pattern: {pattern}")
                continue
            
            pattern_data['first_pos_in_text'] = pattern_data['positions_in_text'].apply(lambda x: x[0] if x else -1)
            pattern_data = pattern_data[pattern_data['first_pos_in_text'] >= 0]
            
            scatter = ax.scatter(
                pattern_data['position_in_rule'], 
                pattern_data['first_pos_in_text'],
                alpha=0.6, 
                s=100,
                c=pattern_data['position_in_rule'],
                cmap='viridis'
            )
            
            if len(pattern_data) > 2:
                z = np.polyfit(pattern_data['position_in_rule'], pattern_data['first_pos_in_text'], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(pattern_data['position_in_rule'].min(), pattern_data['position_in_rule'].max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope={z[0]:.2f})')
            
            ax.set_xlabel("Position in Rule (0 = First)")
            ax.set_ylabel("Character Position in Generated Text")
            ax.set_title(f"Pattern: {pattern}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_rule_position_vs_text_position.png", dpi=300)
        plt.close()
    
    def plot_follow_rate_by_component_type(self):
        """Bar plot: Follow rate by component type (coherent vs random)"""
        def get_component_sequence(pattern):
            if '|' in pattern:
                return pattern.split('|')
            else:
                return list(pattern)
        
        self.expanded_df['component_type'] = self.expanded_df.apply(
            lambda row: get_component_sequence(row['pattern'])[row['position_in_rule'] % len(get_component_sequence(row['pattern']))],
            axis=1
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        component_follow_rate = self.expanded_df.groupby('component_type')['found'].agg(['sum', 'count']).reset_index()
        component_follow_rate['follow_rate'] = component_follow_rate['sum'] / component_follow_rate['count']
        component_follow_rate = component_follow_rate.sort_values('follow_rate', ascending=False)
        
        colors = ['green' if ct == 'c' or ct.startswith('c') else 'red' for ct in component_follow_rate['component_type']]
        ax.barh(component_follow_rate['component_type'], component_follow_rate['follow_rate'], color=colors, alpha=0.7)
        ax.set_xlabel("Follow Rate")
        ax.set_title("Rule Following Rate by Component Type (Coherent vs Random)")
        ax.set_xlim(0, 1)
        
        for i, (ct, rate) in enumerate(zip(component_follow_rate['component_type'], component_follow_rate['follow_rate'])):
            ax.text(rate + 0.02, i, f'{rate:.2%}', va='center')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_follow_rate_by_component_type.png", dpi=300)
        plt.close()
    
    def plot_primacy_recency_bias(self):
        """Line plots: Primacy & Recency bias analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        self.expanded_df['position_quintile'] = pd.qcut(self.expanded_df['position_in_rule'], q=5, duplicates='drop', labels=['1st', '2nd', '3rd', '4th', '5th'])
        
        quintile_follow = self.expanded_df.groupby('position_quintile')['found'].agg(['sum', 'count']).reset_index()
        quintile_follow['follow_rate'] = quintile_follow['sum'] / quintile_follow['count']
        
        axes[0].bar(range(len(quintile_follow)), quintile_follow['follow_rate'], color='steelblue', alpha=0.7)
        axes[0].set_xticks(range(len(quintile_follow)))
        axes[0].set_xticklabels(quintile_follow['position_quintile'])
        axes[0].set_ylabel("Follow Rate")
        axes[0].set_title("Primacy & Recency: Follow Rate by Position Quintile")
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, rate in enumerate(quintile_follow['follow_rate']):
            axes[0].text(i, rate + 0.02, f'{rate:.2%}', ha='center')
        
        pattern_quintile = self.expanded_df.groupby(['pattern', 'position_quintile'])['found'].mean().unstack()
        pattern_quintile.plot(kind='bar', ax=axes[1], alpha=0.8)
        axes[1].set_ylabel("Follow Rate")
        axes[1].set_title("Follow Rate by Position Quintile (per Pattern)")
        axes[1].set_ylim(0, 1)
        axes[1].legend(title='Position Quintile', bbox_to_anchor=(1.05, 1))
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_primacy_recency_bias.png", dpi=300)
        plt.close()
    
    def create_all(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        self.plot_follow_rate_by_position_pattern()
        print("✅ Follow rate by position & pattern")
        self.plot_rule_position_vs_text_position()
        print("✅ Rule position vs text position")
        self.plot_follow_rate_by_component_type()
        print("✅ Follow rate by component type")
        self.plot_primacy_recency_bias()
        print("✅ Primacy & recency bias")
        print(f"📊 All visualizations saved to {self.output_dir}")