"""Shared ReVSI scoring and frame-budget handling.

Scoring protocol: https://github.com/eamonn-zh/torchmetrics_ext/blob/main/src/torchmetrics_ext/metrics/vqa/revsi.py
Frame indices: https://huggingface.co/datasets/3dlg-hcvc/ReVSI
"""

import json
import math
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from loguru import logger

MRA = "MRA:.5:.95:.05"
MCA_QUESTION_TYPES = (
    "object_rel_direction_forward_easy",
    "object_rel_direction_backward_easy",
    "object_rel_direction_forward_hard",
    "object_rel_direction_backward_hard",
    "object_rel_distance_closest",
    "object_rel_distance_farthest",
    "route_planning",
)
NA_QUESTION_TYPES = (
    "object_counting_single",
    "object_counting_multiple",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation_single",
    "room_size_estimation_multiple",
)
GROUPS = {
    "object_rel_direction": MCA_QUESTION_TYPES[:4],
    "object_rel_distance": MCA_QUESTION_TYPES[4:6],
    "object_counting": NA_QUESTION_TYPES[:2],
    "room_size_estimation": NA_QUESTION_TYPES[-2:],
}


def process_docs(dataset):
    if os.getenv("LMMS_EVAL_SHUFFLE_DOCS"):
        return dataset.shuffle(seed=42)
    return dataset


def first_token(response):
    tokens = str(response or "").strip().split()
    return tokens[0].rstrip(".") if tokens else ""


def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    try:
        pred, target = float(pred), float(target)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(pred) or not math.isfinite(target) or target <= 0:
        return 0.0
    # Preserve the official evaluator's threshold construction and comparison.
    thresholds = np.linspace(start, end, int((end - start) / interval + 2))
    return float((abs(pred - target) / target <= 1 - thresholds).mean())


def process_results(doc, results, metric_name):
    record = dict(doc)
    prediction = results[0] if results else ""
    record["prediction"] = prediction
    question_type = record["question_type"]
    if question_type in MCA_QUESTION_TYPES:
        record["accuracy"] = float(first_token(prediction).lower() == str(record["ground_truth"]).lower())
    elif question_type in NA_QUESTION_TYPES:
        record[MRA] = mean_relative_accuracy(first_token(prediction), record["ground_truth"])
    else:
        raise ValueError(f"Unknown ReVSI question type: {question_type}")
    return {metric_name: record}


def summarize_results(results):
    """Average each subtype, merge four families, then average seven categories.

    For partial runs, an incomplete family remains NaN and is omitted from the
    overall mean, matching the author's evaluator. Use a full split for reporting.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty ReVSI evaluation.")
    scores = defaultdict(list)
    for record in results:
        question_type = record["question_type"]
        if question_type in MCA_QUESTION_TYPES:
            metric = "accuracy"
        elif question_type in NA_QUESTION_TYPES:
            metric = MRA
        else:
            raise ValueError(f"Unknown ReVSI question type: {question_type}")
        value = float(record[metric])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid ReVSI score for {question_type}: {value}")
        scores[question_type].append(value)

    output = {}
    for question_type in MCA_QUESTION_TYPES + NA_QUESTION_TYPES:
        values = scores[question_type]
        output[question_type] = sum(values) / len(values) if values else math.nan
    for category, subtypes in GROUPS.items():
        output[category] = sum(output.pop(subtype) for subtype in subtypes) / len(subtypes)
    available = [score for score in output.values() if math.isfinite(score)]
    output["overall"] = sum(available) / len(available) if available else math.nan
    return output


def aggregate_results(results):
    output = summarize_results(results)
    percentages = {key: value * 100 for key, value in output.items()}
    logger.info(f"ReVSI evaluation results (0–100): {percentages}")
    return output["overall"] * 100


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt") or "These are frames of a video."
    question_type = doc["question_type"]
    if question_type in NA_QUESTION_TYPES:
        post_prompt = kwargs.get("na_post_prompt") or "Answer using a single number."
        return "\n".join((pre_prompt, doc["question"], post_prompt))
    if question_type in MCA_QUESTION_TYPES:
        post_prompt = kwargs.get("mca_post_prompt") or "Answer with the option's letter from the given choices directly."
        options = "Options:\n" + "\n".join(doc["options"])
        return "\n".join((pre_prompt, doc["question"], options, post_prompt))
    raise ValueError(f"Unknown ReVSI question type: {question_type}")


@lru_cache(maxsize=4)
def _read_frame_indices(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def frame_indices(doc, frame_budget, video_length):
    """Fixed subsets use the released indices; all-frame uses uniform sampling."""
    if video_length <= 0:
        raise ValueError(f"Empty ReVSI video: {doc['scene_id']}")
    expected_subset = "all" if frame_budget == 128 else str(frame_budget)
    subset = str(doc.get("num_frames", "")).removesuffix("_frame").removesuffix("-frame")
    if subset != expected_subset:
        raise ValueError(f"ReVSI_{frame_budget}frame requires num_frames={expected_subset!r}, got {subset!r}.")
    if frame_budget == 128:
        return np.linspace(0, video_length - 1, min(frame_budget, video_length), dtype=int).tolist()

    root = Path(os.getenv("REVSI_VIDEO_DIR", "data/media/ReVSI"))
    path = Path(os.getenv("REVSI_FRAME_INDEX_PATH", str(root / "metadata/sampled_video_frame_idx.json")))
    if not path.is_file():
        raise FileNotFoundError(f"ReVSI frame indices not found: {path}. Download metadata/sampled_video_frame_idx.json from 3dlg-hcvc/ReVSI or set REVSI_FRAME_INDEX_PATH.")
    indices = _read_frame_indices(str(path))[doc["scene_id"]][f"{frame_budget}-frame"]
    if len(indices) != frame_budget or any(type(index) is not int or not 0 <= index < video_length for index in indices):
        raise ValueError(f"Invalid {frame_budget}-frame indices for {doc['scene_id']} (video length {video_length}).")
    return indices


def doc_to_visual(doc, frame_budget):
    # Return one nested image sequence, as expected by the GeoThinker adapter.
    # This keeps the task's frame budget independent of max_num_frames.
    import decord
    from PIL import Image

    root = Path(os.getenv("REVSI_VIDEO_DIR", "data/media/ReVSI"))
    video_path = root / f"{doc['scene_id']}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"ReVSI video not found: {video_path}. Set REVSI_VIDEO_DIR to the extracted video directory.")
    reader = decord.VideoReader(str(video_path))
    indices = frame_indices(doc, frame_budget, len(reader))
    frames = [Image.fromarray(reader[index].asnumpy()).convert("RGB") for index in indices]
    return [frames]
