# VLM4D Evaluation

## Update: Category Annotations Are Now Available

We have updated the VLM4D evaluation annotations in our [Hugging Face evaluation directory](https://huggingface.co/lihy285/GeoThinker/tree/main/evaluation/VLM4D). Please use the updated [`test.jsonl`](https://huggingface.co/lihy285/GeoThinker/blob/main/evaluation/VLM4D/test.jsonl) with the evaluation adapter in this directory.

When we previously confirmed that VLM4D evaluation was supported, we overlooked a dependency on category information maintained in our local data preparation pipeline. Our local setup included this information, but the earlier public setup did not provide everything needed to reproduce our category-wise evaluation directly. We apologize for the oversight and the extra work this caused.

Question–answer pairs are sufficient to calculate overall accuracy. Reproducing the category-wise breakdown also requires assigning each question to the appropriate category. Our earlier preparation relied on the benchmark's grouping and sample-order conventions to make these assignments locally, rather than making the assignments explicit in each released record.

The updated annotations include a `type` field for each question, making the category assignments used by our adapter available without requiring users to reconstruct our local mapping.

## Annotation Format

For example:

```json
{
  "id": "validation_1",
  "video": "https://huggingface.co/datasets/shijiezhou/VLM4D/resolve/main/videos_real/davis/aerobatics.mp4",
  "question_type": "multiple-choice",
  "question": "What direction is the bus moving towards?",
  "choices": {
    "A": "not moving",
    "B": "there are no buses in the video",
    "C": "right",
    "D": "left"
  },
  "answer": "there are no buses in the video",
  "type": "real_exo",
  "dataset": "VLM4D"
}
```

Here, `type: "real_exo"` identifies a real-world, exocentric sample. The current annotation file contains the category labels `real_exo`, `real_ego`, `directional`, and `FP`.

The `type` field provides the grouping information for category-wise aggregation. It is separate from `question_type`, which describes the question format, such as `multiple-choice`.

## Using the Updated Annotations

1. Download the updated `test.jsonl` linked above and place it under `data/evaluation/VLM4D/`, replacing any older local annotation file. Avoid keeping older JSON/JSONL annotations in the same loading directory.
2. Obtain the corresponding videos from the [original VLM4D dataset](https://huggingface.co/datasets/shijiezhou/VLM4D), preserving their relative paths under `data/media/VLM4D/`. For example, the video above should resolve to `data/media/VLM4D/videos_real/davis/aerobatics.mp4` with the default adapter paths.
3. Check [VLM4D.yaml](VLM4D.yaml). When loading the published `test.jsonl` through the local JSON dataset loader, use:

   ```yaml
   dataset_path: data/evaluation/VLM4D
   test_split: test
   ```

   The sample ID prefix `validation_` does not determine the dataset split. If your configuration still specifies `test_split: validation`, update it to `test` for this file layout, or explicitly configure a matching split mapping in your own setup.

4. Run the repository's evaluation workflow with `--tasks VLM4D`. See the [main evaluation instructions](../../../../README.md#evaluation) for model and launcher configuration.

The current adapter calculates overall accuracy and logs category-wise accuracy by grouping records using `type`. Overall accuracy is returned as a percentage, while category values in the log are fractions between 0 and 1. Category-wise values are currently logged rather than returned as separate top-level metrics.

If you previously downloaded the annotations, please refresh your local copy. The updated records remove the need to reconstruct our category assignments from sample positions. Thank you to the users who brought this missing dependency to our attention.
