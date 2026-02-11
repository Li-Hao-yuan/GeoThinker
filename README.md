# Thinking with Geometry: Active Geometry Integration for Spatial Reasoning

<div align="center" margin-bottom="3em">
<a href="https://arxiv.org/abs/2602.06037" target="_blank">
<img src="https://img.shields.io/badge/arXiv-GeoThinker-green" alt="arXiv"></a>
<a href="https://li-hao-yuan.github.io/GeoThinker/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/Website-GeoThinker-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/lihy285/GeoThinker" target="_blank">
    <img alt="Model" src="https://img.shields.io/badge/Model-GeoThinker-yellow.svg" height="20" />
</a>

</div>
&nbsp

<div align="center" margin-bottom="3em">
Haoyuan Li<sup>*</sup>, Qihang Cao<sup>*</sup>, Tao Tang, Kun Xiang, Zihan Guo, Jianhua Han, Hang Xu, Xiaodan Liang<sup>&ddagger;</sup>

<sup>*</sup>Equal contribution.
<sup>&ddagger;</sup> Corresponding author.

</div>
&nbsp;

Recent progress in spatial reasoning with Multimodal Large Language Models (MLLMs) increasingly leverages geometric priors from 3D encoders. However, most existing integration strategies remain passive: geometry is exposed as a global stream and fused in an indiscriminate manner, which often induces semantic-geometry misalignment and redundant signals.
We propose GeoThinker, a framework that shifts the paradigm from passive fusion to active perception. 
Instead of feature mixing, GeoThinker enables the model to selectively retrieve geometric evidence conditioned on its internal reasoning demands.

<p align="center">
    <img src="assets/Teaser_v7.png" width="100%"><br>
</p>

## 📢News
* [2026-02-04] We release our model weight.
* [2026-02-02] We release our code.


## ✨Architecture Overview

GeoThinker represents a paradigm shift in how Multimodal Large Language Models (MLLMs) understand the 3D world, moving from <span style="text-decoration: underline;">passive fusion</span> to <span style="text-decoration: underline;">active perception</span>. GeoThinker empowers the model to selectively query the geometric information it needs to solve a specific task.
1. Dual-Encoder Processing

    The system begins by processing input video frames through two specialized paths. 
    * A 2D vision encoder extracts high-level semantic features (the "what" of the scene). 
    * A 3D visual geometry encoder (VGGT) captures fine-grained spatial structures and inter-frame dependencies (the "where" and "how" of the 3D space).

2. Spatial-Grounded Fusion (SGF)

    Instead of mixing features at the start, GeoThinker uses the Spatial-Grounded Fusion (SGF) module to manage interactions. 
    * SGF employs frame-strict cross-attention, ensuring that visual tokens only look at geometric cues from the corresponding frame to maintain spatial consistency. 
    * An Importance Gating module predicts a localized bias, allowing the model to focus on salient geometric features like object boundaries while ignoring redundant noise like empty floors or walls.



<p align="center">
    <img src="assets/Model_v7.png" width="100%"><br>
    <figcaption align="center">The architecture of our GeoThinker.</figcaption>
</p>


## 🚀Main Results Highlights
* **Spatial Reasoning (VSI-Bench):** Archieves an average score of 72.6%.
    <p align="center">
        <img src="assets/vsi_bench.png" width="100%"><br>
    </p>

* **Spatial Reasoning (EASI-leaderboard):** Archieves an average score of 55.0%, ranked by 6.

    <p align="center">
        <img src="assets/easi_leaderboard.png" width="100%"><br>
    </p>

## ⚙️Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Li-Hao-yuan/GeoThinker](https://github.com/Li-Hao-yuan/GeoThinker)
    cd GeoThinker
    ```

2.  **Create a Conda environment and install dependencies:**
    We recommend using Python 3.10.
    ```bash
    conda create -n geothinker python=3.10
    conda activate geothinker
    pip install -e .
    ```


## 📊Datasets

GeoThinker is trained and evaluated on a variety of datasets:

* **Vanilla Regime:**
    * [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M): We used a subset of ~234K samples (3% of original). Data prep follows official codebase, navigation type discarded.
    * [LLaVA-Video-178K (LLaVA-Hound split)](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K): We used a subset of ~63K samples (25% of original). 
    * Evaluation Benchmarks: We adopt [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench) for main evaluation.
* **Scaled Regime:**
    * [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M): As mentioned at **Vanilla Regime** above.
    * [LLaVA-Video-178K (LLaVA-Hound split)](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K): As mentioned at **Vanilla Regime** above.
    * [VLM-3R](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA): We leverage instruction data of VSI-Bench and VSTI-Bench for fine-tuning. Notably, we excluded ARkitScene annotations to ensure all data align with the corresponding images available in the ScanNet subset of the SPAR-7M dataset.
    * [PhysGame](https://huggingface.co/PhysGame/PhysVLM-SFT): We utilize the PhysVLM-SFT data from the PhysGame dataset. For this specific subset, we sampled 8 frames per video. (Increasing this to 32 frames may further enhance model performance)
    * [VSI-590k](https://huggingface.co/datasets/nyu-visionx/VSI-590K): We alos utilize the VSI-590k instruction dataset, sourced from Cambrian-S.
    * [Cambrian-s-3M](https://huggingface.co/datasets/nyu-visionx/Cambrian-S-3M): To enhance general video understanding, we incorporate a 430k subset (approximately 11%) of the Cambrian-S-3M dataset. This diverse mixture includes various sources such as <span><i>ActivityNet, CLEVR, FAVD, GUIWorld, K400, K710, LSMDC, MovieChat, NextQA, ShareGPT4Video, SSV2, STAR, TextVR, TGIF, TimeIT, TVQA, VidLN, WebVid</i>, and <i>YouCook2</i></span>.
    * Evaluation Benchmarks: We adopt [VSI-Bench](https://huggingface.co/datasets/nyu-visionx/VSI-Bench), [MMSI-Bench](https://huggingface.co/datasets/RunsenXu/MMSI-Bench), [MindCube](https://huggingface.co/datasets/MLL-Lab/MindCube), [ViewSpatial](https://huggingface.co/datasets/lidingm/ViewSpatial-Bench), [SITE](https://huggingface.co/datasets/franky-veteran/SITE-Bench), [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench) and etc. for evaluation.

## Finetuned Models

Our models are built upon two variants of Qwen2.5-VL, [Qwen2.5-VL—3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct),  [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and are integrated with VGGT-1B as the 3D geometry encoder. 
We recommend adopting the 7B model as the backbone, as our experiments show it delivers both superior performance and faster inference speed.

| | Model Access |
|---|---|
| Vanilla Regime | [🤗GeoThinker-Qwen2.5VL-3B](https://huggingface.co/lihy285/GeoThinker/tree/main/GeoThinker-VGGT-Vanilla-Qwen25VL-3B)<br>[🤗GeoThinker-Qwen2.5VL-7B](https://huggingface.co/lihy285/GeoThinker/tree/main/GeoThinker-VGGT-Vanilla-Qwen25VL-7B) |
| Spatial Reasoning | [🤗GeoThinker-Qwen2.5VL-7B](https://huggingface.co/lihy285/GeoThinker/tree/main/GeoThinker-VGGT-Scaled-Qwen25VL-7B)<br>[🤗GeoThinker-Qwen3VL-8B](https://huggingface.co/lihy285/GeoThinker/tree/main/GeoThinker-VGGT-Scaled-Qwen3VL-8B) |


## Data Preparation

### 1. Structure
Before starting the training process, you need to download the required datasets and annotations according to the following folder structure.

<details>
<summary>data structure.</summary>

```
data
|-- datasets
|   `-- VLM4D
|-- evaluation
|   |-- GameBench
|   |-- MVBench
|   |-- MindCube
|   |-- OmniSpatial
|   |-- PhyBlock
|   |-- QuantiPhy_v
|   |-- SITE
|   |-- SPAR_Bench
|   |-- SR3D
|   |-- VLM4D
|   |-- VSI-Debiased
|   |-- VSI-Super
|   |-- VSTI-bench
|   |-- VideoMMMU
|   |-- VideoMME
|   |-- ViewSpatial
|   `-- embspatial
|-- media
|   |-- BLINK
|   |-- CV_Bench
|   |-- ChartQA
|   |-- GameBench
|   |-- MMSI_Bench
|   |-- MVBench
|   |-- MindCube
|   |-- PhyBlock
|   |-- QSpatial_plus
|   |-- QSpatial_scannet
|   |-- SITE
|   |-- SR3D
|   |-- VSI-Bench
|   |-- VSTI-bench
|   |-- ViewSpatial
|   |-- ai2d
|   |-- llava_hound
|   |-- ocrbench
|   `-- spar
`-- train
    |-- cambrian_s_3m_clean_16frame_obs.json
    |-- cambrian_s_3m_clean_32frame_obs.json
    |-- llava_hound_255k.json
    |-- llava_hound_64k.json
    |-- llava_hound_64k_16frame.json
    |-- llava_hound_64k_32frame.json
    |-- mindcube_10k.json
    |-- phygames_140k.json
    |-- spar_234k.json
    |-- vlm3r_vsi_205k.json
    |-- vlm3r_vsi_205k_16frames.json
    |-- vlm3r_vsi_205k_32frames.json
    |-- vlm3r_vst_132k.json
    |-- vlm3r_vst_132k_16frames.json
    |-- vlm3r_vst_132k_32frames.json
    |-- vsi_590k.json
    |-- vsi_590k_16frame_obs.json
    `-- vsi_590k_32frame_obs.json
```
</details>

### 2. Data for Vanilla Regime
  * **Annotations:** Download the annotation files from [VG-LLM-Data](https://huggingface.co/datasets/zd11024/VG-LLM-Data).
  * **Video Data:** Download the media data of LLaVA-Video-178K (LLaVA-Hound split) from the [ShareGPTVideo](https://huggingface.co/datasets/ShareGPTVideo/train\_video\_and\_instruction/tree/main/train\_300k).
  * **SPAR Data:** Download the media data of SPAR from [SPAR-7M](https://huggingface.co/datasets/jasonzhango/SPAR-7M).

### 3. Data for Scaled Regime
* **Annotations:** Download the annotation files from [GeoThinker-train](https://huggingface.co/lihy285/GeoThinker/tree/main/train).
  * **LLaVA-Hound&SPAR:** As mentioned at **Vanilla Regime** above.
  * **VLM-3R:** The media data from SPAR can be utilized for VLM-3R VSI and VSTI instruction data.
  * **VSI-590k:** Download the media data from [VSI-590k](https://huggingface.co/datasets/nyu-visionx/VSI-590K).
  * **PhysGame:** Download the media data from [PhysGame](https://huggingface.co/PhysGame/PhysVLM-SFT).
  * **Cambrian-s-3M-subset**: Download the selected media data from [Cambrian-s-3M](https://huggingface.co/datasets/nyu-visionx/Cambrian-S-3M). The selected media data can be found at [here](./data/data_links.txt). However, we only used a portion of the data, so you may have some data that was not utilized.

We have provided two example entries as follows
<details>
<summary>Example for LLaVA-Video-178K (LLaVA-Hound Split).</summary>

```json
{
    "id": "23230678_1",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nWhat is the contrast provided in the video's midway point?"
        },
        {
            "from": "gpt",
            "value": "In the midway point of the video, a handgun is displayed on a surface covered with documents, providing a stark contrast to the earlier images of the cigarette being inhaled."
        }
    ],
    "data_source": "llava_hound",
    "video": "llava_hound/frames/23230678"
}
```

</details>

<details>
<summary>Example for SPAR-7M.</summary>

```json
{
    "id": "scene0012_01_1661",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<image>\n<image>\nAssume the depth of box (red point) is 2.0. How much deeper or shallower is chair (green point) relative to table (blue point), measured in meters? Calculate or judge based on the 3D center points of these objects. The depth is calculated based on the image where the markers corresponding to these objects are located. Provide a numeric response with just one value."
        },
        {
            "from": "gpt",
            "value": "1.5"
        }
    ],
    "images": [
        "spar/scannet/images/scene0012_01/image_color/2626.jpg",
        "spar/scannet/images/scene0012_01/image_color/3321.jpg",
        "spar/scannet/images/scene0012_01/image_color/133.jpg"
    ],
    "spar_info": "{\"red_point\": [[395, 89]], \"blue_point\": [[494, 620]], \"green_point\": [[878, 737]], \"point_img_idx\": [[0, 2, 1]], \"type\": \"depth_prediction_oo_mv\"}"
}
```

</details>


### 4. Configure Data Paths

Next, you need to configure the data paths in the source code. Modify the `src/qwen_vl/data/__init__.py` file to ensure the script can locate your datasets.

  * `annotation_path`: This should point to the JSON or JSONL file containing your downloaded dataset annotations.
  * `data_path`: This can be left empty if the image and video paths specified in your annotation files are absolute paths. Otherwise, provide the directory where your data is stored.

## Training

We train two models separately for 3D scene understanding and spatial reasoning tasks. The following instructions are for 3D scene understanidng.

For spatial reasoning, run the following command:

```bash
# Qwen2.5VL backbone
bash scripts/train/train_sr_qwen25vl.sh

# Qwen3VL backbone
bash scripts/train/train_sr_qwen3vl.sh
```

#### Training Details
  * **Backbones**: Our models are built upon two sizes of Qwen2.5-VL and Qwen3-VL, and integrated with VGGT-1B as the 3D geometry encoder.
  * **Hardware:** Our experiments were conducted on a setup with **8x NVIDIA H800 (80G)** GPUs.
  * **Hyperparameters:** We trained the model for one epoch using the Adam optimizer with a batch size of 64, a warmup ratio of 0.03, and a learning rate of 1e-5.
  * **Frozen Components:** During training, the visual encoder of the MLLM, the 3D geometry encoder, and the multimodal connector are kept frozen.
  * **Training Duration:**
      * Vanilla Regime: Approximately 9 hours for 8B model.
      * Scaled Regime: Approximately a week for 8B model.

#### Training Notes & Precautions
  * **Transformer version**: There is a version mismatch between the requirements for Qwen3-VL and Qwen2.5-VL backbones:
      * Qwen3-VL: Training the Qwen3-VL backbone requires a more recent version of the transformers. We recommend using transformers==4.57.0.
      * Qwen2.5-VL: This backbone is incompatible with the version required for Qwen3-VL. For training Qwen2.5-VL, please use transformers==4.50.0.
      * Evaluation: For the sake of consistency during inference and benchmarking, transformers==4.57.0 can be used for both models.
  * **Video Data Processing**: In our implementation, we observed that directly loading and decoding video files during training can lead to inefficient memory management and potential overhead. To optimize throughput and stability, we recommend pre-sampling videos into images uniformly prior to training.
  * **Flash Attention in SGF**: We have integrated a Flash Attention implementation for the Spatial-Grounded Fusion (SGF) module. Please note that while Flash Attention accelerates training, we have observed a performance discrepancy between the Flash Attention implementation and the default attention mechanism in certain scenarios.

<!-- The training scripts will be released soon. -->
## Evaluation

Evaluation is performed using the [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) with greedy sampling for generation. For video benchmarks, 32 frames are uniformly sampled for VSI-Bench.

Please refer to the example evaluation script (`scripts/evaluation/eval.sh`) below for detailed command usage. You may need to adjust `model_path`, `benchmark`, or other parameters based on your specific setup and requirements.
```bash
set -e
export LMMS_EVAL_LAUNCHER="accelerate"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NVLS_ENABLE=0
benchmark=vsibench # choices: [vsibench, cvbench, blink_spatial]
output_path=logs/$(TZ="Asia/Shanghai" date "+%Y%m%d")
model_path=QeoThinker-VGGT-Qwen3VL-8B-Scaled

accelerate launch --num_processes=8 -m lmms_eval \
    --model geothinker \
    --model_args pretrained=$model_path,use_flash_attention_2=true,max_num_frames=32,max_length=12800,max_pixels=451584,min_pixels=12544 \
    --tasks ${benchmark} \
    --batch_size 1 \
    --output_path $output_path
```

## 📋Todo List

- [x] Release the model weights.
- [x] Release the evaluation code, preprocessing data and training scripts.

## Citation
If you find our work useful, please consider citing:

```bibtex
@article{li2026thinking,
  title={Thinking with Geometry: Active Geometry Integration for Spatial Reasoning},
  author={Haoyuan, Li and Qihang, Cao and Tao, Tang and Kun, Xiang and Zihan, Guo and Jianhua, Han and JiaWang, Bian and Hang, Xu and Xiaodan, Liang},
  year={2026}
}
```

## Acknowledgements


* We thanks [VG-LLM](https://github.com/LaVi-Lab/VG-LLM) for the awesome code base.
* This work is built upon excellent previous research, including [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), [VGGT](https://github.com/facebookresearch/vggt), [SPAR-7M](https://github.com/fudan-zvg/spar), [LLaVA-Video-178K](https://github.com/LLaVA-VL/LLaVA-NeXT), and various 3D datasets like [ScanNet](https://github.com/ScanNet/ScanNet), [VLM-3R](https://huggingface.co/datasets/Journey9ni/VLM-3R-DATA), [VSI-590k](https://huggingface.co/datasets/nyu-visionx/VSI-590K), and [Cambrian-s-3M](https://huggingface.co/datasets/nyu-visionx/Cambrian-S-3M).
* We thank the developers of [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for their evaluation framework.