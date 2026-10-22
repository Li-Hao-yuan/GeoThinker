
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import re
import datasets


with open(Path(__file__).parent / "VLM4D.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]
media_dir = dataset_path
# media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]

def vlm4d_doc_to_visual(doc):
    # import pdb; pdb.set_trace()
    video_path = doc["video"].split("resolve/main/")[1]
    video_path = os.path.join(media_dir,video_path).replace("evaluation/", "media/")
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist")
    # import pdb; pdb.set_trace()
    return [video_path]



def vlm4d_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]
    # optionized_list = [f"{key}: {value}" for i, (key, value) in enumerate(doc['choices'].items())]

    options = []
    for key in doc['choices'].keys():
        options.append(key+"."+" "+str(doc['choices'][key]))
    options = "\nOptions:\n" + "\n".join(options)

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    
    # import pdb; pdb.set_trace()

    return pre_prompt + "\n" + question +options+ "\n" + post_prompt

def fuzzy_matching(pred):
    answer_matches = re.findall(r'<answer>(.*?)</answer>', pred, re.DOTALL)
    if len(answer_matches) > 0: pred = answer_matches[-1]

    return pred.split(' ')[0].rstrip('.').strip()

def vlm4d_process_results(doc, results):

    # import pdb; pdb.set_trace()
    
    doc['prediction'] = fuzzy_matching(results[0])
    answer = doc['answer']

    # print(doc['prediction'])

    if doc['prediction'] not in doc['choices']:
        doc["result"] = 0.0
    else:
        doc["result"] = float(answer == doc['choices'][doc['prediction']])

    return {"vlm4d_score": doc}

def vlm4d_aggregate_results(results):

    # import pdb; pdb.set_trace()

    rows = [item["result"] for item in results]
    acc = sum(rows) / len(rows) * 100
    
    output = {}

    results = pd.DataFrame(results)
    for question_type, question_type_indexes in results.groupby('type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        output[f"{question_type}"] = per_question_type['result'].mean()
    
    eval_logger.info(f"Evaluation results: {output}")
    

    return {"accuracy": acc}