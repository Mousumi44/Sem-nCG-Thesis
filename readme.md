

# Sem-nCG Evaluation for Abstractive Summarization

This project adapts the **Sem-nCG** (Semantic Normalized Cumulative Gain) metric for evaluating **abstractive summaries**, specifically using the **CNN/DailyMail** dataset.


## Setup

1. Create a virtual environment (Windows):
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

- **Dataset:** CNN/DailyMail (abstractive)
- Download stories from: https://cs.nyu.edu/~kcho/DMQA/

## Evaluation Metric

- **Sem-nCG@k** = Normalized Cumulative Gain for top-k model sentences

## Wiki and Project Notes

Use a GitHub Wiki or local notes for:

- Meeting minutes
- Dataset versions
- Experiment logs (model configs, results)
- Paper links and highlights
- TODOs and research goals

## Credits

- Sem-nCG: Based on Findings of ACL 2022 paper: https://aclanthology.org/2022.findings-acl.122
- Sentence embeddings: https://www.sbert.net/
- Data annotations: SummEval https://github.com/Yale-LILY/SummEval

## Contact

Project by:
Supervisor by:
Email:

For questions or collaboration, open an issue or contact via email.
