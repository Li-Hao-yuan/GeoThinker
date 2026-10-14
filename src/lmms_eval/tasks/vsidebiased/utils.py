"""VSI-Bench-Debiased shares the original VSI-Bench scoring protocol."""

import os
from pathlib import Path

from lmms_eval.tasks.vsibench.utils import (
    process_docs,
    vsibench_aggregate_results,
    vsibench_doc_to_text,
    vsibench_process_results,
)

vsidebiased_doc_to_text = vsibench_doc_to_text
vsidebiased_aggregate_results = vsibench_aggregate_results


def vsidebiased_doc_to_visual(doc):
    root = Path(os.getenv("VSI_BENCH_VIDEO_DIR", "data/media/VSI-Bench"))
    path = root / doc["dataset"] / f"{doc['scene_name']}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"VSI-Bench video not found: {path}")
    return [str(path)]


def vsidebiased_process_results(doc, results):
    return {"vsidebiased_score": vsibench_process_results(doc, results)["vsibench_score"]}
