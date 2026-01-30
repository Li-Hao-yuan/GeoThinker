
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import re
import datasets
from PIL import Image
import io

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "VideoMME.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]


cache_dir = dataset_path

def VideoMME_doc_to_visual(doc):
    video_path = os.path.join(cache_dir.replace("evaluation", "media"), doc['videoID']+".mp4")
    return [video_path]

def VideoMME_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    options = "Options:\n" + "\n".join(doc["options"])

    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Please answer the question using a single word or phrase."

    return "\n".join([pre_prompt, question, options, post_prompt])

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    answer_matches = re.findall(r'<answer>(.*?)</answer>', pred, re.DOTALL)
    if len(answer_matches) > 0: pred = answer_matches[-1]

    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
}

def VideoMME_process_results(doc, results):

    if "image" in doc.keys(): doc.pop("image")

    doc['prediction'] = results[0]
    for key, value in METRICS_FOR_MCA.items():
        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['answer'])

    return {"VideoMME_score": doc}

def VideoMME_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for class_anno, class_anno_indexes in results.groupby('domain').groups.items():
        per_class_anno = results.iloc[class_anno_indexes]
        
        for metric in METRICS_FOR_MCA.keys():
            output[f"{class_anno}_{metric}".lower()] = per_class_anno[metric].sum()
    
    output['overall'] = sum([_ for _ in output.values()]) #/ len(output)
    output['overall_mean'] = output['overall'] / 2700
    
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall']
