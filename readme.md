sample.json file contains the sample you want to evaluate

format is below:

sample.json -> [{"Doc": str, "Reference": str, "model": str},...]

##To Run the Code
pip install - r requirements.txt

python pre-run.py

python compute_score.py

You will get a score.jsonl file which contains the desired sem-nCG@3 score

##This code has been tested with python 3.7