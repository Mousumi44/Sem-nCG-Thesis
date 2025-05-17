import jsonlines
import json
import pandas as pd
import ast

k = 3

def read_gt_gain():
    """
    :rtype: List[List[tuple]] --> 252 length
    """
    dir = "./output/"
    fn = dir+"stsb_distilbert_gain.txt"
    
    gt_gain = []
    with open(fn, "r") as file:
        for line in file:
            gt_gain.append(ast.literal_eval(line))
    return gt_gain

def read_model_file(file="./output/model.json"):
    """
    :rtype: List[List[int]]
    """
    data = json.load(open(file, "r"))
    df = pd.DataFrame.from_dict(data) #orient="index" must be provided if the json file is in indexed format
    return df["modelSenID"]


def scoreNCG(model_relevance, GT_relevance):
  """
  model_relevance: List[gain] -> for model
  GT_relevance: List[gain] -> ideal
  """

  # CG score 
  cg = sum(model_relevance)

  # ICG score 
  icg = sum(GT_relevance)

  # Normalized CG score 
  ncg = cg/icg

  return ncg

def computeNCG(gt, model):
    """
    #For a sample
    :type gt: List[tuple]
    :type model: List[int]
    :rtype: float -> ncg score
    """
    gt_dic = {int(k): v for k, v in dict(gt).items()}
    gt_rel = [v for _, v in gt[:k]]

    # model_rel = [gt_dic[id] for id in model]

    model_rel = []
    for j in range(k):
        model_rel.append(gt_dic[model[j]]) #take k sentences for model
    return scoreNCG(model_rel, gt_rel)

def eval_ncg():
    gt_gain = read_gt_gain()
    model_senId = read_model_file()
    res = []
    for i in range(len(gt_gain)):
        if model_senId[i] is None:
            print(f"[SKIP] Sample {i}: model_senId is None")
            continue
        if len(model_senId[i]) < k:
            print(f"[SKIP] Sample {i}: model_senId has less than k elements")
            continue
        score = computeNCG(gt_gain[i], model_senId[i])
        if score is not None:
            res.append(score)
    # key_ = args.embed+"_"+args.type+"_lambda_"+str(args.lambda_)
    key_ = "Sem-nCG@3"
    return {key_:res}

if __name__=='__main__':
    with jsonlines.open("score.jsonl", "a") as writer:   # for writing
        writer.write(eval_ncg())