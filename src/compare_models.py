import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import os

# Load the nDCG scores
df = pd.read_csv('output/nDCG_scores.csv')

# Load ROUGE/BERTScore scores if available
rouge_bert_scores = pd.DataFrame()
if os.path.exists('output/rouge_bert_scores.csv'):
    rouge_bert_scores = pd.read_csv('output/rouge_bert_scores.csv')
    print(f"Loaded ROUGE/BERTScore scores: {rouge_bert_scores.shape}")
    
    # Combine with nDCG scores
    if not rouge_bert_scores.empty:
        # Ensure same number of rows
        max_rows = max(len(df), len(rouge_bert_scores))
        if len(df) < max_rows:
            df = df.reindex(range(max_rows))
        if len(rouge_bert_scores) < max_rows:
            rouge_bert_scores = rouge_bert_scores.reindex(range(max_rows))
        
        # Combine datasets
        df_combined = pd.concat([df, rouge_bert_scores], axis=1)
        print(f"Combined scores shape: {df_combined.shape}")
    else:
        df_combined = df
else:
    df_combined = df
    print("ROUGE/BERTScore data not found, using only nDCG scores")

selected_models = [
    
    #LLMs

    # 'olmo_abstractive',
    # 'openelm_abstractive',
    # 'llama3.2_abstractive',
    'gemma',
    # 'mistral_abstractive',
    # 'qwen_abstractive',
    # 'selene-llama_abstractive',
    # 'yulan-mini_abstractive',
    # 'falcon_abstractive'


    ## classical models
    'senticse', 
    'roberta', 
    'sbert-mini', 
    'simcse', 
    'use',
     
    # ROUGE/BERTScore metrics
    'rouge1', 'rouge2', 'rougeL', 'bertscore'
]

# Filter to models that actually exist in the combined data
available_models = [model for model in selected_models if model in df_combined.columns]
print(f"Available models for visualization: {available_models}")

df_selected = df_combined[available_models]

# Create visualizations with combined data
model_means = df_selected.mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=model_means.index, y=model_means.values, palette='viridis')

# Add data labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)

plt.xticks(rotation=45, ha='right')
plt.title("Average Scores (nDCG + ROUGE/BERTScore)")
plt.ylabel("Mean Score")
plt.tight_layout()
plt.savefig('output/scores_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_selected)
plt.xticks(rotation=45, ha='right')
plt.title("Distribution of Scores per Model (nDCG + ROUGE/BERTScore)")
plt.tight_layout()
plt.savefig('output/scores_box_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Load Kendall's Tau results from both sources
kendall_df = pd.read_csv('output/kendall_tau_ndcg_results.csv')

# Load ROUGE/BERTScore Kendall results if available
rouge_bert_kendall = pd.DataFrame()
if os.path.exists('output/rouge_bert_kendall_results.csv'):
    rouge_bert_kendall = pd.read_csv('output/rouge_bert_kendall_results.csv')
    # Rename 'metric' to 'model' for consistency if needed
    if 'metric' in rouge_bert_kendall.columns:
        rouge_bert_kendall = rouge_bert_kendall.rename(columns={'metric': 'model'})
    print(f"Loaded ROUGE/BERTScore Kendall results: {rouge_bert_kendall.shape}")
    
    # Combine Kendall results
    kendall_df_combined = pd.concat([kendall_df, rouge_bert_kendall], ignore_index=True)
    print(f"Combined Kendall results shape: {kendall_df_combined.shape}")
else:
    kendall_df_combined = kendall_df
    print("ROUGE/BERTScore Kendall data not found, using only nDCG Kendall results")

# Filter Kendall results to selected models
kendall_df_selected = kendall_df_combined[kendall_df_combined['model'].isin(available_models)]

print(f"Kendall data for selected models: {len(kendall_df_selected)} rows")

# Create correlation table like the research paper format
print("\n" + "="*80)
print("HUMAN EVALUATION CORRELATION TABLE (Kendall's Tau)")
print("="*80)

if not kendall_df_selected.empty:
    correlation_table = kendall_df_selected.pivot_table(
        values='tau', 
        index='model', 
        columns='annotation', 
        aggfunc='mean'
    ).round(2)
    
    # Add average column
    correlation_table['Average'] = correlation_table.mean(axis=1).round(2)
    correlation_table = correlation_table.sort_values('Average', ascending=False)
    
    # Print formatted table
    header = f"{'Metric':<25}"
    for col in correlation_table.columns:
        header += f"{col:>12}"
    print(header)
    print("-" * len(header))
    
    for metric, row in correlation_table.iterrows():
        row_str = f"{metric:<25}"
        for value in row:
            if pd.isna(value):
                row_str += f"{'N/A':>12}"
            else:
                row_str += f"{value:>12.2f}"
        print(row_str)
    
    # Save correlation table
    correlation_table.to_csv('output/correlation_table.csv')
    print(f"\nCorrelation table saved to output/correlation_table.csv")

print("="*80)

# Create Kendall visualizations with combined data
if not kendall_df_selected.empty:
    heatmap_data = kendall_df_selected.pivot_table(
        values='tau', index='model', columns='annotation', aggfunc='mean'
    )

    plt.figure(figsize=(12, 8))
    mask = heatmap_data.isnull()
    sns.heatmap(heatmap_data, annot=True, cmap="vlag", center=0, fmt=".2f", 
               mask=mask, square=True, linewidths=0.5)
    plt.title("Kendall's Tau Correlation: Models vs Annotations (Combined)")
    plt.tight_layout()
    plt.savefig('output/kendall_heat_map.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(14, 6))
    ax = sns.barplot(data=kendall_df_selected, x="model", y="tau", hue="annotation", palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title("Kendall's Tau by Model and Annotation (Combined)")
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('output/kendall_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Add significance if p-values are available
    if 'p' in kendall_df_selected.columns:
        kendall_df_selected['significant'] = kendall_df_selected['p'] < 0.05

        plt.figure(figsize=(14, 6))
        sns.scatterplot(
            data=kendall_df_selected, x="model", y="tau", hue="annotation",
            style="significant", s=100, palette="deep"
        )
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title("Kendall's Tau with Significance Highlight (Combined)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('output/kendall_scatter_sig_plot.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Facet grid
    if len(kendall_df_selected) > 0:
        palette_facet = sns.color_palette("Set2", len(kendall_df_selected["model"].unique()))
        g = sns.FacetGrid(kendall_df_selected, col="annotation", col_wrap=2, height=4, sharey=True)
        g.map_dataframe(sns.barplot, x="tau", y="model", hue="model", 
                       order=kendall_df_selected["model"].unique(), palette=palette_facet)
        g.set_titles("{col_name}")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle("Kendall's Tau per Annotation Type (Combined)", fontsize=16)
        plt.savefig('output/kendall_facet_grid.png', dpi=300, bbox_inches='tight')
        plt.close()

print("\nPlots saved in the output/ directory:")
print("  - scores_bar_plot.png (combined scores)")
print("  - scores_box_plot.png (score distributions)")
print("  - kendall_heat_map.png (correlation heatmap)")
print("  - kendall_bar_plot.png (grouped bar chart)")
print("  - kendall_scatter_sig_plot.png (scatter with significance)")
print("  - kendall_facet_grid.png (faceted by annotation)")
print("  - correlation_table.csv (formatted correlation table)")
