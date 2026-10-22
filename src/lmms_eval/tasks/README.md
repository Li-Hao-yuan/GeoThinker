# Evaluation Adapters and Reproduction Notes

This directory contains the benchmark adapters used in our evaluation framework. We also release supporting evaluation annotations in the [GeoThinker evaluation directory](https://huggingface.co/lihy285/GeoThinker/tree/main/evaluation).

These adapters and files include changes made for our model interfaces, local data layout, and experimental settings. They should not be assumed to be exact copies of the official benchmark implementations or complete specifications of every historical evaluation run. Some adaptations reflect our interpretation of the benchmark requirements, and implementation mistakes or missing preparation steps may remain.

We apologize for cases where these assumptions were not sufficiently documented. The notes below explain several important distinctions when using this directory.

## Official Benchmark Protocols

Please consult each benchmark's official annotations, evaluation code, and submission instructions when checking results. If our adapter differs from the official protocol, use the official resources to establish the benchmark's intended evaluation procedure and document any differences in your run.

Successfully completing inference in this framework does not, by itself, establish that a score follows the official protocol. Frame selection, input ordering, prompts, answer parsing, category mappings, and score aggregation can all affect the result.

Task-specific notes, such as the [VLM4D README](VLM4D/README.md), provide additional details where available.

## Frame Budgets and Multi-Video Samples

A frame budget may apply to each video rather than to an entire question. Some tasks also perform frame extraction inside the task adapter, so the model-level `max_num_frames` setting is not necessarily the effective limit for that task.

For example, [MMSI-Video-Bench](https://github.com/InternRobotics/MMSI-Video-Bench) can include multiple videos and reference images in one sample. Our [adapter](MMSI_Video_Bench/utils.py) samples each video separately and combines its frames with the other visual inputs in the sample's listed order.

- With a budget of 32 frames per video, two sufficiently long videos can contribute 64 frames in total, plus any reference images.
- The current adapter defaults to **up to 50 uniformly sampled frames per video**, controlled by `MMSI_VIDEO_MAX_FRAMES`. Two sufficiently long videos can therefore contribute 100 frames, plus reference images.
- This adapter does not implement the official Sufficient-Coverage selection strategy.

The 32-frame example illustrates per-video budgeting; it is not the current MMSI-Video default. Check the task adapter and run configuration before describing an evaluation as using “32 frames per sample.” Current defaults should also not be used to infer the exact settings of an earlier run.

## Annotation and Media Paths

Our local setup generally separates evaluation annotations from media:

```text
data/
├── evaluation/
│   └── <benchmark>/    # Annotation files
└── media/
    └── <benchmark>/    # Videos and images
```

Some adapters derive media paths from the annotation directory, including by replacing `evaluation` with `media`. Others use task-specific directories or environment variables. Historical runs also involved local path adjustments to accommodate dataset-loading behavior and avoid problems caused by placing media and annotations in the same loading directory.

These conventions are specific to this framework, rather than requirements of the original benchmarks. When preparing a task, check both its YAML configuration and its media-loading function. Updating `dataset_path` alone may not update every media path. Also ensure that the configured dataset split matches the files being loaded.

## EgoSchema: Prediction Generation and Official Scoring

**Local scores computed against placeholder answers are not valid EgoSchema benchmark results.**

For our full-set EgoSchema inference preparation, we used a constant placeholder answer in the local annotation records so that the framework could run its usual prediction pipeline despite the absence of full-set ground-truth answers. These placeholders are not model predictions and are not genuine answer labels.

The [local adapter](egoschema/utils.py) compares predictions with the `answer` field provided in the loaded records. It does not identify whether those values are placeholders. Consequently, it may print an apparent accuracy even when the records contain dummy answers. That value only measures agreement with the placeholders and must not be reported as model performance.

For full-benchmark evaluation:

1. Run inference and save the per-question predictions with their original question identifiers.
2. Convert the predictions into the format required by the official submission procedure, checking the answer-option encoding.
3. Submit them through an evaluation route described in the [official EgoSchema instructions](https://github.com/egoschema/EgoSchema#benchmarking-on-egoschema).
4. Use the official evaluation result as the full-benchmark score.

The official release also includes answers for a **500-question public subset**. Local accuracy can be meaningful on that subset when using the genuine labels and matching question identifiers, but it is a subset result, not a full-benchmark result. Consult the official instructions for available submission routes and current requirements.

## Reporting Results and Issues

When comparing results, record the annotation version and split, effective frame budget per video and per sample, any input-processing changes, and whether the score came from local ground-truth evaluation or an official submission service.

If you encounter a discrepancy, please open an issue with the task name, code revision, annotation source, relevant configuration, and a reproducible example. References to the official implementation are particularly helpful for checking and correcting adaptations.
