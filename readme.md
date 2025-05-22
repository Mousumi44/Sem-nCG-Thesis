\section*{Sem-nCG Evaluation for Abstractive Summarization}

This project adapts the \textbf{Sem-nCG} (Semantic Normalized Cumulative Gain) metric for evaluating \textbf{abstractive summaries}, specifically using the \textbf{CNN/DailyMail} dataset.

Originally designed for extractive summaries, Sem-nCG is extended here with \textbf{semantic sentence alignment} to handle the paraphrasing and compression commonly found in abstractive summaries.

\subsection*{Project Structure}

\begin{verbatim}
Sem-nCG/
├── pre-run.py                # Preprocessing: computes similarity and gain
├── compute_score.py          # Calculates Sem-nCG@k using gain files and model summaries
├── compute_senID()           # Rewritten for semantic sentence alignment using SBERT
├── sample.json               # Input JSON: doc, reference, model summary
├── output/
│   ├── stsb_distilbert.txt       # Sentence similarity scores
│   ├── stsb_distilbert_gain.txt  # Ranked sentence gains
│   └── model.json                # Model summaries and sentence alignments
├── requirements.txt
├── README.md
└── wiki/                    # (Optional) GitHub wiki for notes, dataset, experiments
\end{verbatim}

\subsection*{Setup}

\begin{enumerate}
  \item Create a virtual environment (Windows):
  \begin{verbatim}
  python -m venv venv
  venv\Scripts\activate
  \end{verbatim}
  
  \item Install dependencies:
  \begin{verbatim}
  pip install -r requirements.txt
  \end{verbatim}
\end{enumerate}

\subsection*{Dataset}

\begin{itemize}
  \item Dataset: \textbf{CNN/DailyMail} (abstractive)
  \item Download stories from: \url{https://cs.nyu.edu/~kcho/DMQA/}
  \item Use \texttt{data\_processing/pair\_data.py} from the SummEval repo:
    \url{https://github.com/Yale-LILY/SummEval}
\end{itemize}

\subsection*{Evaluation Metric}

\begin{itemize}
  \item \textbf{Sem-nCG@k} = Normalized Cumulative Gain for top-k model sentences
  \item Semantic similarity computed via SBERT: \texttt{all-MiniLM-L6-v2}
  \item Model summary sentences aligned to document sentences via cosine similarity
  \item Document sentences are ranked by similarity to reference summary to define gains
\end{itemize}

\subsection*{Wiki and Project Notes}

Use a GitHub Wiki or local notes for:

\begin{itemize}
  \item Meeting minutes
  \item Dataset versions
  \item Experiment logs (model configs, results)
  \item Paper links and highlights
  \item TODOs and research goals
\end{itemize}

\subsection*{Credits}

\begin{itemize}
  \item Sem-nCG: Based on \textit{Findings of ACL 2022} paper: \url{https://aclanthology.org/2022.findings-acl.122}
  \item Sentence embeddings: \url{https://www.sbert.net/}
  \item Data annotations: SummEval \url{https://github.com/Yale-LILY/SummEval}
\end{itemize}

\subsection*{Contact}

Project by:
Email:

For questions or collaboration, open an issue or contact via email.
