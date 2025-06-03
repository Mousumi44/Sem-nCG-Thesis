

# Sem-nCG Evaluation for Abstractive Summarization

This project adapts the **Sem-nCG** (Semantic Normalized Cumulative Gain) metric for evaluating **abstractive summaries**, specifically using the **CNN/DailyMail** dataset.

## Meeting notes

Link: https://docs.google.com/document/d/14Vnd6cudsuq0BEn5OQaOyqqRJqkU4Ov87VTqEmlWU1w/edit?usp=sharing

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


## Credits

- Sem-nCG: Based on Findings of ACL 2022 paper: https://aclanthology.org/2022.findings-acl.122
- Data annotations: SummEval https://github.com/Yale-LILY/SummEval

## Contact

- Project by: Muhammad Farmal Khan (farmal.khan@tu-dortmund.de)
- Supervised by: Mousoumi Akter (mousumi.akter@tu-dortmund.de)

For questions or collaboration, open an issue or contact via email.
