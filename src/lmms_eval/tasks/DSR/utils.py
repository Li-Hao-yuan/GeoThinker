"""DSR task for the local annotations prepared by GeoThinker-F."""

import os
from collections import defaultdict
from pathlib import Path
from loguru import logger


def DSR_doc_to_visual(doc):
    root = Path(os.getenv("DSR_VIDEO_DIR", "data/media/DSR"))
    name = str(doc["videoID"]).rsplit("_", 1)[0] + ".mp4"
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"DSR video not found: {path}")
    return [str(path)]


def DSR_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt") or "These are frames of a video."
    post_prompt = kwargs.get("mca_post_prompt") or "Answer with the option's letter from the given choices directly."
    return "\n".join((pre_prompt, doc["question"], "Options:\n" + "\n".join(doc["options"]), post_prompt))


def DSR_process_results(doc, results):
    record = dict(doc)
    record["prediction"] = results[0] if results else ""
    tokens = str(record["prediction"] or "").strip().split()
    answer = tokens[0].rstrip(".").lower() if tokens else ""
    record["accuracy"] = float(answer == str(record["answer"]).lower())
    return {"DSR_score": record}


def DSR_aggregate_results(results):
    if not results:
        raise ValueError("Cannot aggregate an empty DSR evaluation.")
    categories = defaultdict(list)
    for record in results:
        categories[record["type"]].append(record["accuracy"])
    output = {category: sum(scores) / len(scores) for category, scores in categories.items()}
    # Preserve the GeoThinker-F task's average over question types.
    output["overall"] = sum(output.values()) / len(output)
    logger.info(f"DSR evaluation results: {output}")
    return output["overall"] * 100
