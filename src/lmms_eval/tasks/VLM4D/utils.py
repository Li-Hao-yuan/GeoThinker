
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets


with open(Path(__file__).parent / "vlm4d.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

# dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]

media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]

def vlm4d_doc_to_visual(doc):
    video_path = doc["video"]
    video_path = os.path.join(media_dir,video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist")
    return [video_path]



def vlm4d_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    # optionized_list = [f"{key}: {value}" for i, (key, value) in enumerate(doc['choices'].items())]
    optionized_str = doc['choices'][0].strip()[1:-1].replace(", ", ",\n")
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    
    return pre_prompt + "\n" + question +optionized_str+ "\n" + post_prompt
    
import ast
def vlm4d_process_results(doc, results):
    
    doc['prediction'] = results[0]
    answer = doc['answer']

    choices_str = doc['choices'][0]
    choices_dict = ast.literal_eval(choices_str)  # 安全地把字符串转成 dict
    predict = choices_dict.get(doc['prediction'], None)
    doc["result"] = answer == predict

    return {"vlm3d_score": doc}

def vlm4d_aggregate_results(results):

    rows = [item["vlm3d_score"] for item in results]

    df = pd.DataFrame(rows)

    # 统计正确数
    correct = df["result"].sum()   # 因为 result 是 True/False，可以直接 sum
    acc = correct / len(df)

    return {"accuracy": acc, "details": df}