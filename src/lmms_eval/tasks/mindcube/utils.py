
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image
import datasets

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

yaml_path = Path(__file__).parent / "mindcube.yaml"
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

def mindcube_doc_to_visual(doc):
    # import pdb
    # pdb.set_trace()

    image_files = doc["images"]
    for i, image_file in enumerate(image_files):
        image_files[i] = os.path.join(cache_dir, image_file).replace("evaluation", "media")
    images = [
        Image.open(image_file).convert("RGB") for image_file in image_files
    ]
    return [images]


def mindcube_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    # import pdb
    # pdb.set_trace()

    pre_prompt = "These are frames of a video." 
    question = doc["question"]
    post_prompt = "Answer with the option's letter from the given choices directly."
    return "\n".join([pre_prompt, question, post_prompt])

def fuzzy_matching(text: str) -> str:
    # 只取第一个词，去掉结尾的句点，并做大小写归一
    return (text or "").split(" ")[0].rstrip(".").strip().lower()
def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.
def mindcube_process_results(doc, results):
    doc["prediction"] = results[0]
    for key, value in METRICS_FOR_MCA.items():
        # pred = fuzzy_matching(doc["prediction"])
        # gold = fuzzy_matching(doc["gt_answer"])
        doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc["gt_answer"])      # True 表示对，False 表示错
        
    
    return {"mindcube_score": doc}


def mindcube_aggregate_results(results):

    df = pd.DataFrame(results)
    if "id" not in df.columns:
        raise ValueError("缺少 'id' 列，无法根据前缀分组。")

    # 选一个可用的分数列
    cand_cols = ("acc", "is_correct", "accuracy", "score")
    col = next((c for c in cand_cols if c in df.columns), None)
    if col is None:
        raise ValueError(f"找不到用于计算准确率的列，期望之一：{cand_cols}。现有列：{list(df.columns)}")

    # 取出并规范化到 [0,1]
    vals = df[col]
    if vals.dtype == bool:
        vals = vals.astype(float)
    vals = pd.to_numeric(vals, errors="coerce")
    vals = vals.dropna()
    df = df.loc[vals.index].copy()
    if vals.empty:
        raise ValueError(f"列 '{col}' 没有有效数值。")
    if vals.max() > 1.0:
        vals = vals / 100.0
    df["acc_val"] = vals.clip(lower=0.0, upper=1.0)

    # 从 id 中抽取前缀
    df["category"] = (
        df["id"].astype(str)
        .str.extract(r"^([^_]+)_", expand=False)  # 取第一个下划线前
        .str.lower()
        .fillna("unknown")
    )

    # # 只保留指定三类（如果你想“自动发现全部前缀”，把这行筛选去掉即可）
    # if allowed_prefixes:
    #     df = df[df["category"].isin([p.lower() for p in allowed_prefixes])]
    #     if df.empty:
    #         raise ValueError(f"筛选后无样本。请检查 id 前缀是否在 {allowed_prefixes} 中，或传 allowed_prefixes=None 放开筛选。")

    # 分组计算
    per_cat = df.groupby("category")["acc_val"].mean().to_dict()
    overall = float(df["acc_val"].mean())  # micro 平均（样本加权）

    # 组织输出为百分比
    output = {f"{k}_acc": v * 100.0 for k, v in per_cat.items()}
    output["overall"] = overall * 100.0
    print(output)
    return output