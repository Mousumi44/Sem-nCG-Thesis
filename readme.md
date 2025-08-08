

# Document-aware Evaluation of Abstractive Summaries

This project adapts the **Sem-nCG** (Semantic Normalized Cumulative Gain) metric for evaluating **abstractive summaries**, specifically using the **CNN/DailyMail** dataset.

## Links

* Meeting Logs [link](https://docs.google.com/document/d/14Vnd6cudsuq0BEn5OQaOyqqRJqkU4Ov87VTqEmlWU1w/edit?usp=sharing)
* Weekly Work Progress Presentation [link](https://docs.google.com/presentation/d/1VlFHwL3vgKU85JBegCajIccoRDZgGOrmT_HFhhH1Ewk/edit?usp=sharing )
* Experiment results [link](https://docs.google.com/spreadsheets/d/1T3jnNB1oXLwrRBN1QwlMBgnunM1lM4ZmR3MTr6ZTCtM/edit?gid=0#gid=0)
* Overleaf File [ACL](https://www.overleaf.com/project/685eb4ab7e0aacd38c268ed1) [Thesis](https://www.overleaf.com/project/6881dc3db09af76610e8576d)

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
## Useful References

- [ACL paper](https://aclanthology.org/2022.findings-acl.122)
- [Konvens paper](https://aclanthology.org/2024.konvens-main.21/)
- [SummEval Dataset](https://github.com/Yale-LILY/SummEval) [SummEval Paper](https://aclanthology.org/2021.tacl-1.24.pdf)
  
## Dataset

- **Dataset:** CNN/DailyMail (abstractive)
- Download stories from: https://cs.nyu.edu/~kcho/DMQA/

## Evaluation Metric

- **Sem-nCG@k** = Normalized Cumulative Gain for top-k model sentences




## Contact

- Project by: Muhammad Farmal Khan (farmal.khan@tu-dortmund.de)
- Supervised by: Mousoumi Akter (mousumi.akter@tu-dortmund.de)

For questions or collaboration, open an issue or contact via email.
