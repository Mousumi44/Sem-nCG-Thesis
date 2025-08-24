#!/usr/bin/env python3
"""
Evaluate baseline models (ROUGE, BERTScore, BARTScore) on the processed dataset
Saves scores to CSV for correlation analysis by model_level_correlation.py
"""

import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import os
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate

class BaselineEvaluator:
    def __init__(self, bart_model_name="facebook/bart-large-cnn"):
        """Initialize baseline evaluators"""
        print("Loading baseline models...")
        
        # Initialize ROUGE
        print("  - Loading ROUGE...")
        self.rouge = evaluate.load("rouge")
        
        # Initialize BERTScore
        print("  - Loading BERTScore...")
        self.bertscore = evaluate.load("bertscore")
        
        # Initialize BARTScore
        print("  - Loading BARTScore...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
        self.bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name).to(self.device)
        self.bart_model.eval()
        
        print(f"All models loaded successfully (device: {self.device})")
    
    def compute_rouge_scores(self, references, predictions):
        """Compute ROUGE scores using HuggingFace evaluate"""
        print("Computing ROUGE scores...")
        
        # Compute ROUGE scores
        results = self.rouge.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
            use_aggregator=False  # Get scores for each sample
        )
        
        return {
            'rouge1': results['rouge1'],
            'rouge2': results['rouge2'], 
            'rougeL': results['rougeL'],
            'rougeLsum': results['rougeLsum']
        }
    
    def compute_bertscore(self, references, predictions):
        """Compute BERTScore using HuggingFace evaluate"""
        print("Computing BERTScore...")
        
        # Compute BERTScore with default model (roberta-large)
        results = self.bertscore.compute(
            predictions=predictions,
            references=references,
            lang="en"
        )
        
        # Only return F1 score (the standard BERTScore metric)
        return {
            'bertscore': results['f1'] 
        }
    
    def compute_bartscore(self, source_text, summary_text):
        """
        Compute BARTScore between source and summary
        
        Args:
            source_text: Source document
            summary_text: Generated summary
            
        Returns:
            BARTScore value
        """
        try:
            # Tokenize inputs
            inputs = self.bart_tokenizer(
                source_text,
                summary_text,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get logits from the model
                outputs = self.bart_model(**inputs, labels=inputs["input_ids"])
                
                # BARTScore is negative log-likelihood
                # Lower loss = higher quality
                score = -outputs.loss.item()
                
            return score
            
        except Exception as e:
            print(f"Error computing BARTScore: {e}")
            return 0.0
    
    def compute_bart_scores(self, documents, summaries):
        """Compute BARTScore for all document-summary pairs"""
        print("Computing BARTScore...")
        
        bart_scores = []
        for i, (doc, summary) in enumerate(tqdm(zip(documents, summaries), 
                                               desc="Computing BARTScore", 
                                               total=len(documents))):
            try:
                score = self.compute_bartscore(doc, summary)
                bart_scores.append(score)
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                bart_scores.append(0.0)
        
        return {'bartscore': bart_scores}

def load_data(file_path="./data/processed_data.json", summary_type="Abstractive"):
    """Load and filter data by summary type"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Filter by summary type
    filtered_data = [item for item in data if item.get("summary_type") == summary_type]
    print(f"Loaded {len(filtered_data)} {summary_type} samples")
    
    return filtered_data

def save_baseline_scores(rouge_scores, bert_scores, bart_scores, data, output_dir="./output"):
    """Save baseline model scores to files with model identifiers"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine all scores
    all_scores = {**rouge_scores, **bert_scores, **bart_scores}
    
    # Add model identifiers
    model_ids = [item.get('model_id', f'unknown_{i}') for i, item in enumerate(data)]
    all_scores['model_id'] = model_ids
    
    # Save to CSV with samples as rows and metrics as columns
    df_scores = pd.DataFrame(all_scores)
    # Reorder columns to put model_id first
    cols = ['model_id'] + [col for col in df_scores.columns if col != 'model_id']
    df_scores = df_scores[cols]
    
    scores_file = f"{output_dir}/baseline_models_scores.csv"
    df_scores.to_csv(scores_file, index=False)
    print(f"Baseline model scores saved to {scores_file}")
    
    return all_scores

def main():
    print("BASELINE MODELS EVALUATION")
    print("=" * 50)
    print("Evaluating ROUGE, BERTScore, and BARTScore")
    print("=" * 50)
    
    # Load data
    data = load_data(summary_type="Abstractive")
    
    # Extract texts
    documents = [item["Doc"] for item in data]
    references = [item["Reference"] for item in data]
    model_summaries = [item["model"] for item in data]
    
    print(f"\nEvaluating {len(data)} summary pairs...")
    
    # Initialize baseline evaluator
    evaluator = BaselineEvaluator()
    
    # Compute all baseline scores
    rouge_scores = evaluator.compute_rouge_scores(references, model_summaries)
    bert_scores = evaluator.compute_bertscore(references, model_summaries)
    bart_scores = evaluator.compute_bart_scores(documents, model_summaries)
    
    # Extract human scores (sample-level for statistics)
    print("\nExtracting human annotation scores...")
    # Note: Human scores are now processed by model_level_correlation.py
    
    # Save baseline scores (pass data for model identifiers)
    all_baseline_scores = save_baseline_scores(rouge_scores, bert_scores, bart_scores, data)
    
    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    
    for metric_name, scores in all_baseline_scores.items():
        if metric_name == 'model_id':
            continue
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            print(f"{metric_name:15s}: mean={np.mean(valid_scores):.3f}, "
                  f"std={np.std(valid_scores):.3f}, "
                  f"min={np.min(valid_scores):.3f}, "
                  f"max={np.max(valid_scores):.3f}")
    
    print(f"\nBaseline evaluation complete!")
    print(f"Files generated:")
    print(f"  - ./output/baseline_models_scores.csv (with model_id)")
    print(f"\nTo compute model-level correlations, run:")
    print(f"  python3 src/model_level_correlation.py")

if __name__ == "__main__":
    main()
