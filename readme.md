

# Document-aware Evaluation of Abstractive Summaries

This project adapts the **Sem-nCG** (Semantic Normalized Cumulative Gain) metric for evaluating **abstractive summaries**, specifically using the **CNN/DailyMail** dataset.

## Links

* Meeting Logs [link](https://docs.google.com/document/d/14Vnd6cudsuq0BEn5OQaOyqqRJqkU4Ov87VTqEmlWU1w/edit?usp=sharing)
* Weekly Work Progress Presentation [link](https://docs.google.com/presentation/d/1VlFHwL3vgKU85JBegCajIccoRDZgGOrmT_HFhhH1Ewk/edit?usp=sharing )
* Experiment results [link](https://docs.google.com/spreadsheets/d/1T3jnNB1oXLwrRBN1QwlMBgnunM1lM4ZmR3MTr6ZTCtM/edit?gid=0#gid=0)
* Overleaf File [ACL](https://www.overleaf.com/project/685eb4ab7e0aacd38c268ed1) [Thesis](https://www.overleaf.com/project/6881dc3db09af76610e8576d)

## Setup and Run

## 1. Create and Activate Conda Environment

```bash
conda create -n semncg python=3.11 -y
conda activate semncg
```

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 3. Set Your Hugging Face Token

You need a Hugging Face token to access some models. Get your token from https://huggingface.co/settings/tokens and set it in a `.env` file:

Create a file named `.env` in the project root with this content:

```
HF_TOKEN=your_huggingface_token_here
```

## 4. Run the Application

### Using python

```bash
python src/ndcg_abs.py <MODEL_NAME>
```

### Using shell script

```bash
./run_pipeline.sh <MODEL_NAME>
```

Note: conda has to be activated in both cases

Available models: llama3.2, gemma-3-1b-it, mistral, openelm, olmo-2-1b, qwen3-0.6b, falcon-7b, yulan-mini, sbert-mini, sbert-l, use, roberta, simcse

## 6. Evaluation

To compute correlations and save results in a csv file, run:

```bash
python src/kendall_tau_evaluation.py
```


## 5. Generate Visualizations

After running the main pipeline, generate tables and plots from the output data:

```bash
python src/compare_models.py
```


## 6. Output

Results and plots will be saved in the `output/` and `models/` directories.

---

  
## Dataset

- **Dataset:** CNN/DailyMail (abstractive)
- Download stories from: https://cs.nyu.edu/~kcho/DMQA/

## Evaluation Metric

- **Sem-nCG@k** = Normalized Cumulative Gain for top-k model sentences



## Contact

- Project by: Muhammad Farmal Khan (farmal.khan@tu-dortmund.de)
- Supervised by: Mousoumi Akter (mousumi.akter@tu-dortmund.de)

For questions or collaboration, open an issue or contact via email.
