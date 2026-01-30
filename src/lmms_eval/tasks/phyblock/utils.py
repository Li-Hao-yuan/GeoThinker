
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

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "phyblock.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]


cache_dir = dataset_path

def phyblock_doc_to_visual(doc):
    image_files = doc["images"]
    for i, image_file in enumerate(image_files):
        image_files[i] = os.path.join(cache_dir, image_file).replace("evaluation", "media")
    images = [
        Image.open(image_file).convert("RGB") for image_file in image_files
    ]
    return [images]

def phyblock_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."

    if "<img1>" in doc['options']:
        option_list = ["A","B","C","D","E","F","G"][:len(doc["options"])]
        options = "Options:\n" + "\n".join([a+". <image>\n" for a in option_list])
    else: options = "Options:\n" + "\n".join(doc["options"])

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

def phyblock_process_results(doc, results):
    doc['prediction'] = results[0]
    for key, value in METRICS_FOR_MCA.items():
        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['answer'])

    return {"phyblock_score": doc}

def phyblock_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for class_anno, class_anno_indexes in results.groupby('question_type').groups.items():
        per_class_anno = results.iloc[class_anno_indexes]
        
        for metric in METRICS_FOR_MCA.keys():
            output[f"{class_anno}_{metric}".lower()] = per_class_anno[metric].mean()
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)

    output['object_property'] = sum([
        output['shape_accuracy'],
        output['color_accuracy'],
        output['size_accuracy'],
        output['number_accuracy'],
    ]) / 4.

    output['object_relationship'] = sum([
        output['relativeposition_accuracy'],
        output['relativerotation_accuracy'],
        output['absoluteposition_accuracy'],
        output['dependencies_accuracy'],
    ]) / 4.

    output['scene_understanding'] = sum([
        output['object_accuracy'],
        output['layer_accuracy'],
        output['type_accuracy'],
        output['viewpoint_accuracy'],
    ]) / 4.

    output['dynamic_reasoning'] = sum([
        output['counterfactual_accuracy'],
        output['predictive_accuracy'],
        output['ordering_accuracy'],
        output['affordance_accuracy'],
    ]) / 4.
    
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
