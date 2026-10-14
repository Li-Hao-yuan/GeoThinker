"""Training arguments for the unified GeoThinker method branches.

See the repository LICENSE and NOTICE for licensing scope.
"""

import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    geometry_method: str = field(default="geothinker", metadata={"help": "Geometry branch: geothinker or geothinker_f"})
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    tune_geometry_merger: bool = field(default=False)

    # Geometry encoder configuration
    use_geometry_encoder: bool = field(default=False)  # Whether to use 3D geometry encoder
    geometry_encoder_type: str = field(default="vggt")  # Type of geometry encoder ("vggt", "pi3")
    geometry_encoder_path: str = field(default="facebook/VGGT-1B/")  # Path to pre-trained geometry encoder model
    reference_frame: str = field(default="first")  # Reference frame for geometry encoding ("first", "last"), only available for vggt
    feature_fusion_method: str = field(default="zero")  # Method to fuse geometry and visual features
    fusion_num_layers: int = field(default=1)  # Number of layers in the cross-attention module when feature_fusion_method is "cross_attention"
    geometry_merger_type: str = field(default="mlp")  # Type of geometry feature merger ("mlp", "avg")

    exclude_geometry_encoder_on_save: bool = field(default=False)  # Whether to exclude geometry encoder weights when saving checkpoints/final model

    cam_merger_type: str = field(default="zero")
    add_regressive_token: bool = field(default=False)

    geo_cross_attn: bool = field(default=False)
    geo_inject_version: str = field(default="v2_flash")
    geo_layer_interval: int = field(default=1)
    geo_spatial_bias: bool = field(default=False)
    geo_learn_bias: bool = field(default=False)
    geo_importance_gate: bool = field(default=False)

    use_qwenvl_loss: bool = field(default=False)

    depart_smi_token: bool = field(default=False)
    smi_image_num: int = field(default=8)
    smi_downsample_rate: int = field(default=2)


    force_simulate_vggt: bool = field(default=False)
    predict_next_frame: bool = field(default=False)
    predict_this_geometry: bool = field(default=False)
    predict_next_geometry: bool = field(default=False)
    collect_intermediate_layers: str = field(default="v0")
    llm_per_layer_predict_head: bool = field(default=False)
    llm_per_layer_collect: bool = field(default=False)
    llm_per_layer_predict_to_decoder: bool = field(default=False)
    llm_per_layer_no_geometry_projector: bool = field(default=False)

    loss_text_weight: float = field(default=1)
    loss_image_semantic_weight: float = field(default=1)
    loss_image_geometry_weight: float = field(default=1)
    log_aux_loss_without_backward: bool = field(default=False)

    frozen_predict_head: bool = field(default=False)
    frozen_semantic_predict_head: bool = field(default=False)
    frozen_geometry_predict_head: bool = field(default=False)

