
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


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "gamebench.yaml", "r") as f:
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

def gamebench_doc_to_visual(doc):
    video_path = doc["question_id"]+".mp4"
    video_path = os.path.join(cache_dir, video_path).replace("evaluation","media")
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def gamebench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    options_list = [
        key+"."+" "+doc["options"][key] for key in doc["options"].keys()
    ]
    options = "Options:\n" + "\n".join(options_list)

    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."

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

def gamebench_process_results(doc, results):
    doc['prediction'] = results[0]
    for key, value in METRICS_FOR_MCA.items():
        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['answer'])

    return {"gamebench_score": doc}

def gamebench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for class_anno, class_anno_indexes in results.groupby('subclass_anno').groups.items():
        per_class_anno = results.iloc[class_anno_indexes]
        
        for metric in METRICS_FOR_MCA.keys():
            output[f"{class_anno}_{metric}".replace(" ","_").lower()] = per_class_anno[metric].mean()
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)

    output['material_properties_accuracy'] = sum([
        output['color_accuracy'],
        output['rigidity_accuracy'],
        output['object_shape_accuracy'],
        output['human_body_gesture_accuracy'],
    ]) / 4.

    output['kinematics_accuracy'] = sum([
        output['velocity_accuracy'],
        output['acceleration_accuracy'],
    ]) / 2.

    output['optics_accuracy'] = sum([
        output['absorption_&_transmission_accuracy'],
        output['refraction_accuracy'],
        output['reflection_accuracy'],
    ]) / 3.

    output['mechanics_accuracy'] = sum([
        output['gravity_accuracy'],
        output['elasticity_accuracy'],
        output['friction_accuracy'],
    ]) / 3.
    
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
