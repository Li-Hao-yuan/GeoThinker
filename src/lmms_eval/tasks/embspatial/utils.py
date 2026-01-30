
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
import datasets
import io
import base64

# MCA_QUESTION_TYPES = [
#     "camera_movement_direction",
#     "camera_obj_rel_dist_v1",
#     "camera_obj_rel_dist_v2",
#     "camera_obj_rel_dist_v3",
#     "obj_obj_relative_pos_nf",
#     "obj_obj_relative_pos_ud",
#     "obj_obj_relative_pos_lr",
# ]
# NA_QUESTION_TYPES = [
#     "camera_obj_abs_dist",
#     "camera_displacement",
# ]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

# METRICS_FOR_NA = {
#     "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
# }


# hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
# base_cache_dir = os.path.expanduser(hf_home)
from pathlib import Path
import yaml

yaml_path = Path(__file__).parent / "embspatial.yaml"
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

def embspatial_doc_to_visual(doc):
    # import pdb
    # pdb.set_trace()

    images = [
        Image.open(io.BytesIO(base64.b64decode(doc['image']))).convert("RGB")
    ]
    return [images]


def embspatial_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # import pdb
    # pdb.set_trace()

    question = doc["question"]

    options = doc["answer_options"]
    for i in range(len(options)):
        options[i] = chr(65+i)+"."+" "+options[i]
    options = "Options:\n" + "\n".join(options)

    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    return "\n".join([question, options, post_prompt])

def fuzzy_matching(text: str) -> str:
    # 只取第一个词，去掉结尾的句点，并做大小写归一
    return (text or "").split(" ")[0].rstrip(".").strip().lower()
def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.
def embspatial_process_results(doc, results):
    doc["prediction"] = results[0]
    for key, value in METRICS_FOR_MCA.items():
        # pred = fuzzy_matching(doc["prediction"])
        # gold = fuzzy_matching(doc["gt_answer"])

        answer_dict = {
            0:"A",
            1:"B",
            2:"C",
            3:"D",
        }

        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), answer_dict[doc["answer"]])      # True 表示对，False 表示错
        
    
    return {"embspatial_score": doc}


def embspatial_aggregate_results(results):

    results = pd.DataFrame(results)
    output = {}

    for question_type, question_type_indexes in results.groupby('answer').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        for metric in METRICS_FOR_MCA.keys():
            output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

    output['overall'] = sum([_ for _ in output.values()]) / len(output)

    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.