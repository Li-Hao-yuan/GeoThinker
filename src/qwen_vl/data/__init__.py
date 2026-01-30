import re


SPAR = {
    "annotation_path": "data/train/spar_7m.jsonl",
    "data_path": "data/media",
    "tag": "3d"
}

SPAR_234K = {
    "annotation_path": "data/train/spar_234k.json",
    "data_path": "data/media",
    "tag": "3d"
}

LLAVA_HOUND_255K = {
    "annotation_path": "data/train/llava_hound_255k.json",
    "data_path": "data/media",
    "tag": "2d"
}

LLAVA_HOUND_64K = {
    "annotation_path": "data/train/llava_hound_64k.json",
    "data_path": "data/media",
    "tag": "2d"
}

LLAVA_HOUND_64K_32FRAME = {
    "annotation_path": "data/train/llava_hound_64k_32frame.json",
    "data_path": "data/media",
    "tag": "2d"
}

VLM3R_VSI_205K = {
    "annotation_path": "data/train/vlm3r_vsi_205k.json",
    "data_path": "data/media",
    "tag": "2d"
}

VLM3R_VST_132K = {
    "annotation_path": "data/train/vlm3r_vst_132k.json",
    "data_path": "data/media",
    "tag": "2d"
}

VLM3R_VSI_205K_16FRAMES = {
    "annotation_path": "data/train/vlm3r_vsi_205k_16frames.json",
    "data_path": "data/media",
    "tag": "2d"
}

VLM3R_VST_132K_16FRAMES = {
    "annotation_path": "data/train/vlm3r_vst_132k_16frames.json",
    "data_path": "data/media",
    "tag": "2d"
}

VLM3R_VSI_205K_32FRAMES = {
    "annotation_path": "data/train/vlm3r_vsi_205k_32frames.json",
    "data_path": "data/media",
    "tag": "2d"
}

VLM3R_VST_132K_32FRAMES = {
    "annotation_path": "data/train/vlm3r_vst_132k_32frames.json",
    "data_path": "data/media",
    "tag": "2d"
}


MINDCUBE_10K = {
    "annotation_path": "data/train/mindcube_10k.json",
    "data_path": "data/media",
    "tag": "2d"
}


PHYGAMES_140K = {
    "annotation_path": "data/train/phygames_140k.json",
    "data_path": "data/media",
    "tag": "2d"
}

VSI_590K = {
    "annotation_path": "data/train/vsi_590k.json",
    "data_path": "data/media",
    "tag": "2d"
}

VSI_590K_16FRAME = {
    "annotation_path": "data/train/vsi_590k_16frame.json",
    "data_path": "data/media",
    "tag": "2d"
}

VSI_590K_32FRAME = {
    "annotation_path": "data/train/vsi_590k_32frame.json",
    "data_path": "data/media",
    "tag": "2d"
}

CAMBRIAN_S_3M_SUBSET_16FRAME = {
    "annotation_path": "data/train/cambrian_s_3m_clean_16frame.json",
    "data_path": "data/media",
    "tag": "2d"
}

CAMBRIAN_S_3M_SUBSET_32FRAME = {
    "annotation_path": "data/train/cambrian_s_3m_clean_32frame.json",
    "data_path": "data/media",
    "tag": "2d"
}

data_dict = {

    ## multi-view
    "spar": SPAR,
    "llava_hound_255k": LLAVA_HOUND_255K,
    "spar_234k": SPAR_234K,
    "llava_hound_64k": LLAVA_HOUND_64K,
    "vlm3r_vsi_205k": VLM3R_VSI_205K,
    "vlm3r_vst_132k": VLM3R_VST_132K,
    "mindcube_10k": MINDCUBE_10K,
    "phygames_140k": PHYGAMES_140K,
    "vsi_590k": VSI_590K,

    ## 16 frame
    "vlm3r_vsi_205k_16frames": VLM3R_VSI_205K_16FRAMES,
    "vlm3r_vst_132k_16frames": VLM3R_VST_132K_16FRAMES,
    "vsi_590k_16frame": VSI_590K_16FRAME,
    "cambrian_s_3m_subset_16frame": CAMBRIAN_S_3M_SUBSET_16FRAME,

    ## 32 frame
    "llava_hound_64k_32frame": LLAVA_HOUND_64K_32FRAME,
    "vlm3r_vsi_205k_32frames": VLM3R_VSI_205K_32FRAMES,
    "vlm3r_vst_132k_32frames": VLM3R_VST_132K_32FRAMES,
    "vsi_590k_32frame": VSI_590K_32FRAME,
    "cambrian_s_3m_subset_32frame": CAMBRIAN_S_3M_SUBSET_32FRAME,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    print("dataset_names",dataset_names)
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config["dataset_name"] = dataset_name
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
