import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Load the scores
df = pd.read_csv('output/score.csv')

selected_models = [
    'Sem-nCG@3_distil',
    'Sem-nCG@3_simcse',
    'Sem-nCG@3_roberta_extractive',
    'Sem-nCG@3_openelm',
    'Sem-nCG@3_llama3.2'
]

df_selected = df[selected_models]

# data = df_selected.mean().reset_index()
# data.columns = ['model', 'score']
# categories = data['model'].tolist()
# values = data['score'].tolist()
# values += values[:1]  # close the circle

# # Radar setup
# N = len(categories)
# angles = [n / float(N) * 2 * pi for n in range(N)]
# angles += angles[:1]

# plt.figure(figsize=(8, 6))
# ax = plt.subplot(111, polar=True)
# plt.xticks(angles[:-1], categories, color='grey', size=10)
# ax.plot(angles, values, linewidth=2, linestyle='solid')
# ax.fill(angles, values, alpha=0.25)
# plt.title("Average Sem-nCG@3 Scores Across Models")
# plt.savefig('output/scores_spider_plot.png')
# plt.close()


model_means = df_selected.mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=model_means.index, y=model_means.values)

# Add data labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)

# plt.xticks(rotation=90)
plt.title("Average Sem-nCG@3 Score (Selected Models)")
plt.ylabel("Mean Score")
plt.tight_layout()
plt.savefig('output/scores_bar_plot.png')
plt.close()


plt.figure(figsize=(14, 6))
sns.boxplot(data=df_selected)
# plt.xticks(rotation=90)
plt.title("Distribution of Sem-nCG@3 Scores per Model")
plt.savefig('output/scores_box_plot.png')
plt.close()


# Load Kendall's Tau results
kendall_df = pd.read_csv('output/kendall_results.csv')

selected_models = [
    'Sem-nCG@3_distil',
    'Sem-nCG@3_simcse',
    'Sem-nCG@3_roberta_extractive',
    'Sem-nCG@3_openelm',
    'Sem-nCG@3_llama3.2'
]

kendall_df_selected = kendall_df[kendall_df['model'].isin(selected_models)]

heatmap_data = kendall_df_selected.pivot(index="model", columns="annotation", values="tau")

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap="vlag", center=0, fmt=".2f")
plt.title("Kendall's Tau Correlation: Models vs Annotations")
plt.tight_layout()
plt.savefig('output/kendall_heat_map.png')
plt.close()


plt.figure(figsize=(14, 6))
sns.barplot(data=kendall_df_selected, x="model", y="tau", hue="annotation")
plt.xticks(rotation=90)
plt.title("Kendall's Tau by Model and Annotation")
plt.axhline(0, color='gray', linestyle='--')
plt.tight_layout()
plt.savefig('output/kendall_bar_plot.png')
plt.close()


kendall_df_selected['significant'] = kendall_df_selected['p'] < 0.05

plt.figure(figsize=(14, 6))
sns.scatterplot(
    data=kendall_df_selected, x="model", y="tau", hue="annotation",
    style="significant", s=100, palette="deep"
)
plt.xticks(rotation=90)
plt.axhline(0, color='gray', linestyle='--')
plt.title("Kendall's Tau with Significance Highlight")
plt.tight_layout()
plt.savefig('output/kendall_scatter_sig_plot.png')
plt.close()


g = sns.FacetGrid(kendall_df_selected, col="annotation", col_wrap=2, height=4, sharey=True)
g.map_dataframe(sns.barplot, x="tau", y="model", hue="model", order=kendall_df_selected["model"].unique())
g.set_titles("{col_name}")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Kendall’s Tau per Annotation Type", fontsize=16)
plt.savefig('output/kendall_facet_grid.png')
plt.close()

print("Plots saved in the output/ directory.")
