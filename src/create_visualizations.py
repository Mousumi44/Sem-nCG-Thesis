import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_and_combine_correlation_data():
    """Load and combine nDCG and baseline correlation data"""
    
    # Load nDCG correlations
    ndcg_df = pd.read_csv('output/model_level_correlation_table.csv')
    print(f"Loaded nDCG correlations: {ndcg_df.shape}")
    
    # Load baseline correlations
    baseline_df = pd.read_csv('output/baseline_correlation.csv')
    print(f"Loaded baseline correlations: {baseline_df.shape}")
    
    # Combine both dataframes
    combined_df = pd.concat([ndcg_df, baseline_df], ignore_index=True)
    print(f"Combined correlations shape: {combined_df.shape}")
    
    return combined_df

def load_score_data():
    """Load nDCG and baseline score data for distribution analysis"""
    
    # Load nDCG scores
    try:
        ndcg_scores = pd.read_csv('output/nDCG_scores.csv')
        print(f"Loaded nDCG scores: {ndcg_scores.shape}")
    except FileNotFoundError:
        print("nDCG scores file not found, will skip nDCG distribution plots")
        ndcg_scores = None
    
    # Load baseline scores
    try:
        baseline_scores = pd.read_csv('output/baseline_models_scores.csv')
        print(f"Loaded baseline scores: {baseline_scores.shape}")
    except FileNotFoundError:
        print("Baseline scores file not found, will skip baseline distribution plots")
        baseline_scores = None
    
    return ndcg_scores, baseline_scores

def create_correlation_visualizations():
    """Create visualizations from combined correlation data"""
    
    # Load combined data
    correlation_df = load_and_combine_correlation_data()
    
    # Selected models for visualization
    selected_models = [
        # LLMs
        # 'ndcg_falcon',
        # 'ndcg_llama3.2', 
        # 'ndcg_olmo',
        # 'ndcg_openelm',
        # 'ndcg_gemma',
        # 'ndcg_mistral',
        # 'ndcg_qwen',
        # 'ndcg_selene-llama',
        # 'ndcg_yulan-mini',
        
        # Classical embedding models
        'ndcg_senticse', 
        'ndcg_roberta', 
        'ndcg_sbert-mini', 
        'ndcg_simcse', 
        'ndcg_use',
        'ndcg_sbert-l',
        
        # Baseline metrics
        'rouge1', 
        'rouge2', 
        'rougeL', 
        'bertscore',
        'bartscore'
    ]
    
    # Filter to available models
    available_models = [model for model in selected_models if model in correlation_df['metric'].values]
    correlation_df_filtered = correlation_df[correlation_df['metric'].isin(available_models)]
    
    print(f"Available models for visualization: {len(available_models)}")
    print(f"Models: {available_models}")
    
    # Set up the data for plotting
    correlation_df_filtered = correlation_df_filtered.set_index('metric')
    
    # Sort by average correlation (descending)
    correlation_df_filtered = correlation_df_filtered.sort_values('Average', ascending=False)
    
    # Create bar plot of average correlations
    plt.figure(figsize=(14, 8))
    colors = ['red' if 'ndcg_falcon' in idx else 'skyblue' if 'ndcg_' in idx else 'lightcoral' for idx in correlation_df_filtered.index]
    
    bars = plt.bar(range(len(correlation_df_filtered)), correlation_df_filtered['Average'], color=colors)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(correlation_df_filtered.iterrows()):
        plt.text(i, row['Average'] + 0.01, f'{row["Average"]:.3f}', 
                ha='center', va='bottom', fontsize=9, rotation=0)
    
    plt.xticks(range(len(correlation_df_filtered)), correlation_df_filtered.index, rotation=45, ha='right')
    plt.ylabel('Average Kendall\'s Tau')
    plt.title('Model Performance: Average Correlation with Human Judgments')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    # legend_elements = [
    #     Patch(facecolor='red', label='Falcon (Best nDCG)'),
    #     Patch(facecolor='skyblue', label='Other nDCG Models'),
    #     Patch(facecolor='lightcoral', label='Baseline Models')
    # ]
    # plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('output/combined_correlation_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create heatmap
    plt.figure(figsize=(10, 12))
    
    # Prepare data for heatmap (exclude Average column)
    heatmap_data = correlation_df_filtered[['coherence', 'consistency', 'fluency', 'relevance']]
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlBu_r', center=0, fmt='.3f',
                square=False, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Correlation Heatmap: Models vs Human Judgment Aspects')
    plt.ylabel('Models')
    plt.xlabel('Human Judgment Aspects')
    plt.tight_layout()
    plt.savefig('output/combined_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create grouped bar chart by aspect
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    aspects = ['coherence', 'consistency', 'fluency', 'relevance']
    
    for i, aspect in enumerate(aspects):
        ax = axes[i//2, i%2]
        
        # Sort by this aspect
        aspect_sorted = correlation_df_filtered.sort_values(aspect, ascending=False)
        colors = ['red' if 'ndcg_falcon' in idx else 'skyblue' if 'ndcg_' in idx else 'lightcoral' for idx in aspect_sorted.index]
        
        bars = ax.bar(range(len(aspect_sorted)), aspect_sorted[aspect], color=colors)
        
        # Add value labels
        for j, (idx, val) in enumerate(zip(aspect_sorted.index, aspect_sorted[aspect])):
            ax.text(j, val + 0.01 if val >= 0 else val - 0.03, f'{val:.2f}', 
                   ha='center', va='bottom' if val >= 0 else 'top', fontsize=8)
        
        ax.set_xticks(range(len(aspect_sorted)))
        ax.set_xticklabels(aspect_sorted.index, rotation=45, ha='right')
        ax.set_title(f'{aspect.capitalize()} Correlation')
        ax.set_ylabel('Kendall\'s Tau')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Performance by Human Judgment Aspect', fontsize=16)
    plt.tight_layout()
    plt.savefig('output/combined_correlation_by_aspect.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create comparison table in terminal
    print("\n" + "="*100)
    print("COMBINED HUMAN EVALUATION CORRELATION TABLE (Kendall's Tau)")
    print("="*100)
    
    # Print formatted table
    header = f"{'Metric':<20} {'Coherence':>12} {'Consistency':>12} {'Fluency':>12} {'Relevance':>12} {'Average':>12}"
    print(header)
    print("-" * len(header))
    
    for metric, row in correlation_df_filtered.iterrows():
        row_str = f"{metric:<20}"
        for col in ['coherence', 'consistency', 'fluency', 'relevance', 'Average']:
            if pd.isna(row[col]):
                row_str += f"{'N/A':>12}"
            else:
                row_str += f"{row[col]:>12.3f}"
        print(row_str)
    
    print("="*100)
    
    # Top performers analysis
    print("\n🏆 TOP PERFORMERS:")
    print(f"Overall Best: {correlation_df_filtered.index[0]} (τ = {correlation_df_filtered.iloc[0]['Average']:.3f})")
    print(f"Best Baseline: {correlation_df_filtered[~correlation_df_filtered.index.str.contains('ndcg_')].index[0]}")
    print(f"Best nDCG: {correlation_df_filtered[correlation_df_filtered.index.str.contains('ndcg_')].index[0]}")
    
    print("\n📊 VISUALIZATIONS SAVED:")
    print("  - combined_correlation_bar_plot.png (overall performance)")
    print("  - combined_correlation_heatmap.png (aspect comparison)")
    print("  - combined_correlation_by_aspect.png (detailed by aspect)")

if __name__ == "__main__":
    create_correlation_visualizations()