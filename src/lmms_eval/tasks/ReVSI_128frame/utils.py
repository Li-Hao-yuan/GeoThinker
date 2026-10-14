"""ReVSI 128-frame task; shared official category aggregation."""

from lmms_eval.tasks._task_utils.revsi import (
    aggregate_results,
    doc_to_text,
    doc_to_visual,
    process_docs,
    process_results,
)


def ReVSI_128frame_doc_to_visual(doc):
    return doc_to_visual(doc, 128)


ReVSI_128frame_doc_to_text = doc_to_text
ReVSI_128frame_aggregate_results = aggregate_results


def ReVSI_128frame_process_results(doc, results):
    return process_results(doc, results, "ReVSI_128frame_score")
