
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
import datasets

MCA_QUESTION_TYPES = [
]
NA_QUESTION_TYPES = [
    "horizontal_distance",
    "vertical_distance",
    "1d_horizontal",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(relative_accuracy, delta=2)",
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def relative_accuracy(pred, target, delta=2):

    pred = to_float(pred)
    target = to_float(target)

    if pred is None: return 0.

    if pred >= target/delta and pred <= target*delta:
        return 1.
    else: return 0.


# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)
from pathlib import Path
import yaml

yaml_path = Path(__file__).parent / "SR3D.yaml"
with open(yaml_path, "r", encoding="utf-8") as f:
    raw_data = f.readlines()

safe_data = []
for i, line in enumerate(raw_data):
    if "!function" not in line:
        safe_data.append(line)

dataset_path = yaml.safe_load("".join(safe_data))["dataset_path"]


# if os.path.isdir(dataset_path):
cache_dir = dataset_path
# else:
#     cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]
#     cache_dir = os.path.join(base_cache_dir, cache_name)

def SR3D_doc_to_visual(doc):

    images = [
        Image.open(os.path.join(cache_dir, doc["image_url"]).replace("http://images.cocodataset.org", "coco_images").replace("evaluation","media")).convert("RGB")
    ]

    return [images]


def SR3D_doc_to_text(doc, lmms_eval_specific_kwargs=None):

    question = doc["question"]

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    options = "Options:\n" + "\n".join([key+"."+" "+doc[key] for key in "ABCD" if doc[key] is not None])

    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    return "\n".join([pre_prompt, question, options, post_prompt])

def fuzzy_matching(text: str) -> str:
    # 只取第一个词，去掉结尾的句点，并做大小写归一
    return (text or "").split(" ")[0].rstrip(".").strip().lower()
def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def SR3D_process_results(doc, results):
    doc["prediction"] = results[0]
    for key, value in METRICS_FOR_MCA.items():
        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc["answer"])      # True 表示对，False 表示错
    
    return {"SR3D_score": doc}


def SR3D_aggregate_results(results):

    results = pd.DataFrame(results)
    output = {}

    for question_type, question_type_indexes in results.groupby('category').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        for metric in METRICS_FOR_MCA.keys():
            output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.