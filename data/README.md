# Training Data Preparation and Known Limitations

This document describes the general principles used to prepare our training media, along with known limitations relevant to reproducing the released annotations.

## Why We Use Image Inputs

During training with our framework on H200 GPUs, we observed memory leakage when using the direct video-loading pipeline. We therefore extracted video frames in advance and loaded the resulting images through the paths listed in the training annotations.

This was an engineering decision for our particular framework and training environment, rather than a general claim about video loading on H200 GPUs.

## Media Preparation and Sampling Principles

For sources requiring frame extraction, our general approach was to convert the media to MP4 and extract images before training. When a source already provided images, we used those images directly.

We generally considered two sampling strategies:

- **Fixed-count sampling:** Uniformly select 32 positions across the video, including its beginning and end. For configurations requiring fewer frames, the intended approach was to uniformly subsample the existing sequence—for example, selecting 8 frames from a 32-frame sequence.
- **Time-based sampling:** For selected sources with less temporal variation, sample approximately one frame per second, followed by subsampling if the sequence exceeds the target frame budget. Short videos may therefore contain fewer than 32 images; for example, a video lasting approximately 16 seconds may retain approximately 16 images.

For data such as PhysGame, where important events may occur briefly, our intended preference was fixed-count sampling even for short videos. Denser temporal coverage may help retain transient events, but uniform sampling with a fixed frame budget does not guarantee that every relevant event will be captured.

These are general preparation principles, not a complete specification of every released sample. Actual processing varied across data sources, preparation stages, and historical script versions.

## Known Limitations

### Frame Selection and Ordering

Temporal ordering was not consistently enforced across the historical preprocessing scripts.

Some scripts selected images directly from a filesystem directory listing. Others sorted image paths as strings only after selection. Directory enumeration does not guarantee chronological order, and string sorting does not necessarily match numerical frame order.

Consequently, uniformly selecting positions from such a list does not necessarily produce temporally uniform samples.

The `images` list in a released annotation should therefore be inspected explicitly rather than assumed to be chronological. The training loader inspected in our local code preserves this list order and does not automatically reorder images by timestamp or frame number.

### Question–Answer Consistency After Sampling

Frame extraction and subsampling primarily followed fixed rules. We did not systematically retain question-specific evidence or perform a complete, sample-by-sample revalidation of the original questions and answers after processing.

As a result:

- Objects or events required to answer a question may be absent from the retained images.
- Questions about appearance order or other temporal relationships may be affected by missing frames or nonchronological ordering.
- Positional references such as “the fifth image” may no longer refer to the intended image after selection or reordering.

These limitations may affect the alignment between the visual inputs and the original questions and answers in some samples.

### Incomplete Historical Processing Records

The recovered scripts cover parts of annotation conversion and subsampling, but do not fully document every step from the original media to the candidate image libraries.

Depending on the source, numbers in image filenames may represent candidate-image indices, original video frame indices, or other identifiers. They should not universally be interpreted as decoded frame indices in the original video.

Similarly, the largest image index referenced in the annotations does not necessarily identify the last image in the original candidate library.

## Reproduction and Reprocessing

For strict reproduction of the released data, inspect the corresponding annotations, available scripts, and media together. Extracting the same number of uniformly spaced frames directly from the original video does not guarantee that the resulting images will match the released annotations.

When rebuilding the data, we recommend establishing the mapping between filenames and timestamps, sorting images chronologically before sampling, and checking questions that depend on image positions, temporal order, or transient events. Any corrected dataset should be documented separately from the original release.

We welcome GitHub Issues containing specific sample identifiers, image lists, and reproduction steps to help investigate these cases and improve the documentation.