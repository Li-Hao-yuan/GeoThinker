
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
import re
import datasets


METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "QuantiPhy_v.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]


if os.path.isdir(dataset_path):
    cache_dir = dataset_path
else:
    cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
    cache_dir = os.path.join(base_cache_dir, cache_name)

def QuantiPhy_v_doc_to_visual(doc):
    video_path = os.path.join(cache_dir, doc['video_id']+".mp4").replace("evaluation", "media")
    return [video_path]


def QuantiPhy_v_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
    return pre_prompt + "\n" + question + "\n" + doc['prior'] + "\n" + post_prompt


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

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):

    choice_list = [chr(65+i) for i in range(26)]
    if pred in choice_list and target in choice_list:
        return exact_match(pred, target)
    
    pred = to_float(pred)
    target = to_float(target)

    if pred is None: return 0.

    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def QuantiPhy_v_process_results(doc, results):
    doc['prediction'] = results[0]

    for key, value in METRICS_FOR_NA.items():
        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['answer'])

        try:
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['answer'])
        except TypeError:
            doc[key] = WORST_CASE_FOR_METRICS[key]

    return {"QuantiPhy_v_score": doc}

def QuantiPhy_v_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('video_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        for metric in METRICS_FOR_NA.keys():
            if metric == 'success_rate':
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
            else:
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
