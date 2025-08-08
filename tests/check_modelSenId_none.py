import json
import pandas as pd
from nltk import tokenize

def check_model_senid_none(model_file="./models/model_extractive.json", data_file="./data/processed_data.json"):
    # Load modelSenID file
    with open(model_file, "r") as f:
        model_data = json.load(f)
    model_df = pd.DataFrame.from_dict(model_data)
    model_senids = model_df["modelSenID"]
    models = model_df["model"]

    # Load original data for context
    with open(data_file, "r") as f:
        data = json.load(f)
    data_df = pd.DataFrame.from_dict(data)
    docs = data_df["Doc"]
    refs = data_df["Reference"]

    print(f"Total samples: {len(model_senids)}")
    for i, senid in enumerate(model_senids):
        if senid is None:
            print(f"Sample {i} has modelSenId=None")
            print("--- Document ---")
            print(docs[i])
            print("--- Model Summary ---")
            print(models[i])
            print("--- Reference ---")
            print(refs[i])
            print("---------------------\n")

if __name__ == "__main__":
    check_model_senid_none()
