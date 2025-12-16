import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

sns.set_style("whitegrid")

try:
    df = pd.read_json("joker_task1_retrieval_corpus25_EN.json")
except Exception as e:
    print(f"Error loading JSON file: {e}")
    exit()

print("\n" + "="*80)
print("             EXPLORATORY DATA ANALYSIS (EDA) REPORT - CONSOLE OUTPUT")
print("="*80)

print("\n--- 1. INITIAL INFORMATION AND DATA CLEANING ------------------------------------")
print(f"Initial Dataset Shape: {df.shape}")

print("\n[TABLE 1.1] Missing Values Check (before dropna):")
print(df.isnull().sum().rename('Missing Count').to_frame().to_markdown())

df.dropna(subset=['text'], inplace=True)
print(f"\nShape after handling missing values: {df.shape}")


df['text_length'] = df['text'].astype(str).apply(len)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))
df['unique_word_count'] = df['text'].astype(str).apply(lambda x: len(set(x.split())))

print("\n--- 2. EXAMPLE OF CREATED FEATURES ----------------------------------------------")
print("[TABLE 2.1] First 5 Rows with Textual Features:")
cols_to_display = ['text', 'text_length', 'word_count', 'unique_word_count']
print(df[cols_to_display].head().to_markdown(index=False, numalign="left", stralign="left"))


print("\n--- 3. UNIVARIATE ANALYSIS: DESCRIPTIVE STATISTICS ----------------------------")
stats_df = df[['text_length', 'word_count', 'unique_word_count']].describe(percentiles=[.25, .5, .75, .95]).T
stats_df.index.name = "Variable"
stats_df = stats_df.rename(columns={'50%': 'Median (Q2)', '25%': 'Q1', '75%': 'Q3', '95%': 'P95'})

print("[TABLE 3.1] Central Tendency, Dispersion, and Quartiles:")
print(stats_df[['mean', 'std', 'min', 'Q1', 'Median (Q2)', 'Q3', 'max', 'P95']].to_markdown(numalign="left", stralign="left"))

print("\n[TABLE 3.2] Distribution Shape (Skewness):")
print(df[['text_length', 'word_count', 'unique_word_count']].skew().rename("Skewness").to_frame().to_markdown())

correlation_matrix = df[['text_length', 'word_count', 'unique_word_count']].corr(method='pearson')

print("\n--- 4. MULTIVARIATE ANALYSIS: PEARSON CORRELATION -----------------------------")
print("[TABLE 4.1] Pearson Correlation Matrix:")
print(correlation_matrix.to_markdown(numalign="left", stralign="left"))

print("\n--- 5. VISUALIZATIONS SAVED TO FILES ------------------------------------------")

variables = ['text_length', 'word_count']

for var in variables:
    var_title = var.replace("_", " ").title()
    
    plt.figure(figsize=(8, 4))
    sns.histplot(df[var], bins=50, kde=True, color='skyblue')
    plt.title(f'Distribution of {var_title}')
    plt.xlabel(var_title)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'hist_{var}.png')
    plt.close()
    print(f"Saved: hist_{var}.png")
    
    plt.figure(figsize=(4, 6))
    sns.boxplot(y=df[var], color='lightcoral')
    plt.title(f'Box Plot: Outliers in {var_title}')
    plt.ylabel(var_title)
    plt.tight_layout()
    plt.savefig(f'boxplot_{var}.png')
    plt.close()
    print(f"Saved: boxplot_{var}.png")

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.savefig('heatmap_correlation.png')
plt.close()
print("Saved: heatmap_correlation.png")

plt.figure(figsize=(8, 6))
sns.scatterplot(x='word_count', y='text_length', data=df, color='darkblue', alpha=0.6)
plt.title('Relationship between Word Count and Text Length')
plt.xlabel('Word Count')
plt.ylabel('Text Length (Characters)')
plt.tight_layout()
plt.savefig('scatterplot_word_length.png')
plt.close()
print("Saved: scatterplot_word_length.png")

print("\n" + "="*80)
print("All required printed outputs and PNG files have been generated for documentation.")
print("=========================================================")