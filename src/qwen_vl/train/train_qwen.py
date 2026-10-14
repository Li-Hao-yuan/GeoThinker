# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict, Optional
import shutil
import sys
from pathlib import Path
import numpy as np
import datetime

project_root = Path(__file__).parent.parent.parent
# print("project_root",project_root)
sys.path.insert(0, str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class

# Optional unsafe checkpoint resume bypass for torch<2.6.
# Use only for trusted local checkpoints.
if os.environ.get("HF_ALLOW_UNSAFE_CHECKPOINT_RESUME", "1") == "1":
    try:
        import transformers.trainer as hf_trainer_mod
        import transformers.utils.import_utils as hf_import_utils

        def _allow_unsafe_checkpoint_load():
            return None

        hf_import_utils.check_torch_load_is_safe = _allow_unsafe_checkpoint_load
        hf_trainer_mod.check_torch_load_is_safe = _allow_unsafe_checkpoint_load
        print(
            "[WARN] HF_ALLOW_UNSAFE_CHECKPOINT_RESUME=1: bypassing torch<2.6 checkpoint safety guard for trusted local files."
        )
    except Exception as e:
        print(f"[WARN] Failed to patch checkpoint safety guard: {e}")

from transformers import (
    Qwen2VLForConditionalGeneration,
)
from qwen_vl.data.data_qwen import make_supervised_data_module

from qwen_vl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, AutoProcessor, Qwen2VLImageProcessor, Trainer, AutoConfig, set_seed, enable_full_determinism
from transformers.integrations import TensorBoardCallback

local_rank = None

GEOMETRY_ENCODER_STATE_PREFIXES = (
    "model.geometry_encoder.",
    "geometry_encoder.",
    "module.model.geometry_encoder.",
    "module.geometry_encoder.",
)

from transformers import TrainerCallback
import time
from datetime import timedelta

# This file has been modified for GeoThinker Unified; upstream FastChat-derived license notices remain applicable.

class ETACallback(TrainerCallback):
    def __init__(self):
        self.training_start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        # 在训练正式开始时记录当前时间
        self.training_start_time = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 确保 logs 存在，且已经有了起始时间，且不是第 0 步
        if logs is not None and self.training_start_time is not None and state.global_step > 0:
            # 计算已耗时
            elapsed_time = time.time() - self.training_start_time
            
            steps_done = state.global_step
            steps_total = state.max_steps # 或者使用 state.num_train_epochs * steps_per_epoch
            
            if steps_done > 0:
                avg_time_per_step = elapsed_time / steps_done
                steps_remaining = steps_total - steps_done
                eta_seconds = max(0, steps_remaining * avg_time_per_step)
                
                # 格式化时间并注入 logs
                eta_str = str(timedelta(seconds=int(eta_seconds)))
                logs["eta"] = eta_str

class VisualTensorBoardCallback(TensorBoardCallback):
    """Reuse Trainer TensorBoard writer to log debug images produced by the model."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        control = super().on_log(args, state, control, logs=logs, **kwargs)
        if not state.is_world_process_zero or self.tb_writer is None:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        model_for_events = model.module if hasattr(model, "module") else model

        # Write monitor-only auxiliary losses directly to TensorBoard,
        # without injecting them into Trainer logs.
        breakdown = getattr(model_for_events, "_last_loss_breakdown", None)
        if isinstance(breakdown, dict):
            if breakdown.get("loss_semantic_monitor") is not None:
                self.tb_writer.add_scalar(
                    "loss_semantic_monitor", breakdown["loss_semantic_monitor"], state.global_step
                )
            if breakdown.get("loss_geometry_monitor") is not None:
                self.tb_writer.add_scalar(
                    "loss_geometry_monitor", breakdown["loss_geometry_monitor"], state.global_step
                )
        pop_fn = getattr(model_for_events, "pop_debug_visual_events", None)
        if pop_fn is None:
            self.tb_writer.flush()
            return control

        events = pop_fn()
        if not events:
            return control

        for tag, image_hwc, step in events:
            self.tb_writer.add_image(tag, image_hwc, global_step=step, dataformats="HWC")
        self.tb_writer.flush()
        return control

class LossBreakdownCallback(TrainerCallback):
    """Inject model-provided loss components into Trainer logs."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return control

        model = kwargs.get("model", None)
        if model is None:
            return control

        model_for_logs = model.module if hasattr(model, "module") else model
        breakdown = getattr(model_for_logs, "_last_loss_breakdown", None)
        if not isinstance(breakdown, dict):
            return control

        # Keep names short so they are easy to read in terminal logs.
        if breakdown.get("loss_text") is not None:
            logs["loss_text"] = breakdown["loss_text"]

        if breakdown.get("loss_semantic") is not None:
            logs["loss_semantic"] = breakdown["loss_semantic"]

        if breakdown.get("loss_geometry") is not None:
            logs["loss_geometry"] = breakdown["loss_geometry"]

        return control

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def print_trainable_parameter_summary(model):
    def format_param_count(n: int) -> str:
        return f"{n} params ({n / 1e6:.3f} M)"

    groups = {
        "visual": 0,
        "visual_merger": 0,
        "llm": 0,
        "lm_head": 0,
        "geometry_merger": 0,
        "geometry_encoder": 0,
        "cross_attn": 0,
        "next_frame_head": 0,
        "this_geometry_head": 0,
        "other": 0,
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        n = param.numel()
        if "visual.merger" in name:
            groups["visual_merger"] += n 
        elif ".visual." in name or name.startswith("visual.") or name.startswith("model.visual."):
            groups["visual"] += n
        elif "geometry_merger" in name:
            groups["geometry_merger"] += n
        elif "geometry_encoder" in name:
            groups["geometry_encoder"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n
        elif "next_frame_head" in name:
            groups["next_frame_head"] += n
        elif "this_geometry_head" in name:
            groups["this_geometry_head"] += n
        elif "cross_attn" in name:
            groups["cross_attn"] += n
        elif ".model.language_model." in name or name.startswith("model.language_model."):
            groups["llm"] += n
        elif name.startswith("model."):
            groups["llm"] += n
        else:
            groups["other"] += n

    total = sum(groups.values())
    rank0_print("====                                    ====")
    rank0_print("====  Trainable Param Summary (numel)  ====")
    rank0_print("====                                    ====")
    rank0_print(f"\tTOTAL: {format_param_count(total)}")
    for k in [
        "visual",
        "visual_merger",
        "llm",
        "lm_head",
        "geometry_merger",
        "geometry_encoder",
        "cross_attn",
        "next_frame_head",
        "this_geometry_head",
        "other",
    ]:
        rank0_print(f"\t{k}: {format_param_count(groups[k])}")


OVERRIDABLE_CONFIG_KEYS = [
    "geometry_method",
    "use_geometry_encoder",
    "geometry_encoder_type",
    "geometry_encoder_path",
    "reference_frame",
    "feature_fusion_method",
    "cam_merger_type",
    "training",
    "fusion_num_layers",
    "geometry_merger_type",
    "depart_smi_token",
    "smi_image_num",
    "smi_downsample_rate",
    "predict_next_frame",
    "predict_this_geometry",
    "predict_next_geometry",
    "collect_intermediate_layers",
    "llm_per_layer_predict_head",
    "llm_per_layer_collect",
    "llm_per_layer_predict_to_decoder",
    "llm_per_layer_no_geometry_projector",
    "loss_text_weight",
    "loss_image_semantic_weight",
    "loss_image_geometry_weight",
    "log_aux_loss_without_backward",
    "geo_cross_attn",
    "geo_inject_version",
    "geo_layer_interval",
    "geo_spatial_bias",
    "geo_learn_bias",
    "geo_importance_gate",
    "use_qwenvl_loss",
    "exclude_geometry_encoder_on_save",
    'force_simulate_vggt',
]


def apply_model_arg_overrides_to_config(config, model_args, include_text_config=False):
    for k in OVERRIDABLE_CONFIG_KEYS:
        if hasattr(model_args, k):
            setattr(config, k, getattr(model_args, k))
            if include_text_config and hasattr(config, "text_config"):
                setattr(config.text_config, k, getattr(model_args, k))


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer,
    output_dir: str,
    exclude_geometry_encoder_on_save: bool = False,
):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if exclude_geometry_encoder_on_save:
        state_dict, removed_keys = _strip_geometry_encoder_from_state_dict(state_dict)
        if trainer.args.should_save and removed_keys > 0:
            rank0_print(f"Skip saving geometry encoder params: {removed_keys} keys")

    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _strip_geometry_encoder_from_state_dict(state_dict):
    filtered = {}
    removed_keys = 0

    for key, value in state_dict.items():
        if key.startswith(GEOMETRY_ENCODER_STATE_PREFIXES):
            removed_keys += 1
            continue
        filtered[key] = value

    return filtered, removed_keys


class GeoAwareTrainer(Trainer):
    def _save(self, output_dir: str = None, state_dict=None):
        exclude_geometry_encoder_on_save = bool(
            getattr(self.model, "_exclude_geometry_encoder_on_save", False)
        )

        if not exclude_geometry_encoder_on_save:
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        if state_dict is None:
            state_dict = self.model.state_dict()

        state_dict, removed_keys = _strip_geometry_encoder_from_state_dict(state_dict)
        if removed_keys > 0 and self.args.should_save:
            rank0_print(f"Skip saving geometry encoder params: {removed_keys} keys")

        return super()._save(output_dir=output_dir, state_dict=state_dict)


def set_model(model_args, model):
    def get_llm_named_parameters(mm_model):
        # Qwen3 VL wraps text backbone as model.model.language_model,
        # while Qwen2.5 VL uses model.model directly as text backbone.
        if hasattr(mm_model, "model") and hasattr(mm_model.model, "language_model"):
            return mm_model.model.language_model.named_parameters()
        return mm_model.model.named_parameters()

    QWEN3VL_FLAG = hasattr(model, "model") and hasattr(model.model, "language_model")

    if model_args.tune_mm_vision:
        if QWEN3VL_FLAG:
            for n, p in model.model.visual.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.visual.named_parameters():
                p.requires_grad = True
    else:
        if QWEN3VL_FLAG:
            for n, p in model.model.visual.named_parameters():
                p.requires_grad = False
        else:
            for n, p in model.visual.named_parameters():
                p.requires_grad = False

    if model_args.tune_mm_mlp:
        if QWEN3VL_FLAG:
            if hasattr(model.model.visual, "merger"):
                for n, p in model.model.visual.merger.named_parameters():
                    p.requires_grad = True
        else:
            if hasattr(model.visual, "merger"):
                for n, p in model.visual.merger.named_parameters():
                    p.requires_grad = True
    else:
        if QWEN3VL_FLAG:
            if hasattr(model.model.visual, "merger"):
                for n, p in model.model.visual.merger.named_parameters():
                    p.requires_grad = False
        else:
            if hasattr(model.visual, "merger"):
                for n, p in model.visual.merger.named_parameters():
                    p.requires_grad = False

    if model_args.tune_mm_llm:
        # Qwen2.5VL: Model(model, visual)
        # Qwen3VL: Model.Model(LLm, visual)

        if QWEN3VL_FLAG:
            for n, p in model.model.language_model.named_parameters():
                p.requires_grad = True
        else:
            for n, p in model.model.named_parameters():
                p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in get_llm_named_parameters(model):
            if "cross_attn" in n: p.requires_grad = True
            else: p.requires_grad = False

        use_direct_layer_prediction = bool(
            getattr(model_args, "llm_per_layer_predict_head", False)
        ) and str(getattr(model_args, "collect_intermediate_layers", "v0")) != "v0"
        train_predict_heads = not use_direct_layer_prediction

        if hasattr(model, "model"):

            if hasattr(model, "next_frame_head"):
                for n, p in model.next_frame_head.named_parameters():
                    p.requires_grad = train_predict_heads
            if hasattr(model, "this_geometry_head"):
                for n, p in model.this_geometry_head.named_parameters():
                    p.requires_grad = train_predict_heads

            if hasattr(model.model, "next_frame_head"):
                for n, p in model.model.next_frame_head.named_parameters():
                    p.requires_grad = train_predict_heads
            if hasattr(model.model, "this_geometry_head"):
                for n, p in model.model.this_geometry_head.named_parameters():
                    p.requires_grad = train_predict_heads

            if hasattr(model.model, "per_layer_predict_head"):
                for n, p in model.model.per_layer_predict_head.named_parameters():
                    p.requires_grad = True
            if hasattr(model.model, "per_layer_geometry_predict_head"):
                for n, p in model.model.per_layer_geometry_predict_head.named_parameters():
                    p.requires_grad = True
            # if hasattr(model.model, "per_layer_geometry_to_language"):
            #     for n, p in model.model.per_layer_geometry_to_language.named_parameters():
            #         p.requires_grad = True



            if QWEN3VL_FLAG:

                if hasattr(model.model.language_model, "per_layer_predict_head"):
                    for n, p in model.model.language_model.per_layer_predict_head.named_parameters():
                        p.requires_grad = True
                if hasattr(model.model.language_model, "per_layer_geometry_predict_head"):
                    for n, p in model.model.language_model.per_layer_geometry_predict_head.named_parameters():
                        p.requires_grad = True
                # if hasattr(model.model.language_model, "per_layer_geometry_to_language"):
                #     for n, p in model.model.language_model.per_layer_geometry_to_language.named_parameters():
                #         p.requires_grad = True

        for p in model.lm_head.parameters():
            p.requires_grad = False

    if model_args.frozen_semantic_predict_head or model_args.frozen_predict_head:
        if hasattr(model, "next_frame_head"):
            for n, p in model.next_frame_head.named_parameters():
                p.requires_grad = False

        if hasattr(model, "model"):

            if hasattr(model.model, "next_frame_head"):
                for n, p in model.model.next_frame_head.named_parameters():
                    p.requires_grad = False

            if hasattr(model.model, "per_layer_predict_head"):
                for n, p in model.model.per_layer_predict_head.named_parameters():
                    p.requires_grad = False

            if hasattr(model.model, "language_model"):

                if hasattr(model.model.language_model, "per_layer_predict_head"):
                    for n, p in model.model.language_model.per_layer_predict_head.named_parameters():
                        p.requires_grad = False

    if model_args.frozen_geometry_predict_head or model_args.frozen_predict_head:
        if hasattr(model, "this_geometry_head"):
            for n, p in model.this_geometry_head.named_parameters():
                p.requires_grad = False

        if hasattr(model, "model"):

            if hasattr(model.model, "this_geometry_head"):
                for n, p in model.model.this_geometry_head.named_parameters():
                    p.requires_grad = False

            if hasattr(model.model, "per_layer_geometry_predict_head"):
                for n, p in model.model.per_layer_geometry_predict_head.named_parameters():
                    p.requires_grad = False
            
            # if hasattr(model.model, "per_layer_geometry_to_language"):
            #     for n, p in model.model.per_layer_geometry_to_language.named_parameters():
            #         p.requires_grad = False


            if hasattr(model.model, "language_model"):

                if hasattr(model.model.language_model, "per_layer_geometry_predict_head"):
                    for n, p in model.model.language_model.per_layer_geometry_predict_head.named_parameters():
                        p.requires_grad = False
                
                # if hasattr(model.model.language_model, "per_layer_geometry_to_language"):
                #     for n, p in model.model.language_model.per_layer_geometry_to_language.named_parameters():
                #         p.requires_grad = False


    assert (model_args.use_geometry_encoder and model_args.force_simulate_vggt)==False, "use_geometry_encoder and force_simulate_vggt cannot be True at the same time."
    if model_args.use_geometry_encoder:
        # vggt is frozen
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False

    if hasattr(model, "geometry_merger"):
        tune = getattr(model_args, "tune_geometry_merger", False) or getattr(model_args, "feature_fusion_method", "zero") != "zero"
        for n, p in model.geometry_merger.named_parameters():
            p.requires_grad = tune
    
    if hasattr(model, "model") and hasattr(model.model, "geometry_merger"):
        tune = getattr(model_args, "tune_geometry_merger", False) or getattr(model_args, "feature_fusion_method", "zero") != "zero"
        for n, p in model.model.geometry_merger.named_parameters():
            p.requires_grad = tune
    
    # model.model.language_model.norm.weight.requires_grad = True

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.geometry_method = model_args.geometry_method.lower()
    if model_args.geometry_method not in {"geothinker", "geothinker_f"}:
        raise ValueError(
            "geometry_method must be one of: geothinker, geothinker_f; "
            f"got {model_args.geometry_method!r}"
        )
    if model_args.geometry_method == "geothinker_f":
        model_args.geo_inject_version = "v3_2"

    setattr(model_args, 'training', True)

    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower() or "OUTPUT_QWEN25" in model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        apply_model_arg_overrides_to_config(config, model_args, include_text_config=False)

        if model_args.geometry_method == "geothinker" and not (model_args.use_geometry_encoder or model_args.force_simulate_vggt):
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        else:
            if model_args.geometry_method == "geothinker":
                from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT as ModelClass
            elif model_args.geometry_method == "geothinker_f":
                from qwen_vl.model.modeling_qwen2_5_vl_geothinker_f import Qwen2_5_VLForConditionalGenerationWithVGGT as ModelClass
            else:
                raise ValueError(f"Unsupported geometry_method: {model_args.geometry_method!r}")
            print(model_args.model_name_or_path)

            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            if model_args.use_geometry_encoder or model_args.force_simulate_vggt:
                assert model_args.geometry_encoder_path is not None, \
                    "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = ModelClass.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                geometry_encoder_path=(
                    model_args.geometry_encoder_path
                    if (model_args.use_geometry_encoder or model_args.force_simulate_vggt)
                    else None
                )
            )

            if getattr(config, 'depart_smi_token', False):
                data_args.depart_smi_token = True
                data_args.smi_image_num = getattr(config, 'smi_image_num', 8)
                data_args.smi_downsample_rate = getattr(config, 'smi_downsample_rate', 2)

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        data_args.model_type = "qwen2.5vl"
    elif "qwen3" in model_args.model_name_or_path.lower() or "OUTPUT_QWEN3" in model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        apply_model_arg_overrides_to_config(config, model_args, include_text_config=True)

        if model_args.geometry_method == "geothinker" and not (model_args.use_geometry_encoder or model_args.force_simulate_vggt):
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        else:
            if model_args.geometry_method == "geothinker":
                from qwen_vl.model.qwenvl3.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT as ModelClass
            elif model_args.geometry_method == "geothinker_f":
                from qwen_vl.model.qwenvl3.modeling_qwen3_vl_geothinker_f import Qwen3VLForConditionalGenerationWithVGGT as ModelClass
            else:
                raise ValueError(f"Unsupported geometry_method: {model_args.geometry_method!r}")
            print(model_args.model_name_or_path)
            
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = ModelClass.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                geometry_encoder_path=model_args.geometry_encoder_path
            )

            if getattr(config, 'depart_smi_token', False):
                data_args.depart_smi_token = True
                data_args.smi_image_num = getattr(config, 'smi_image_num', 8)
                data_args.smi_downsample_rate = getattr(config, 'smi_downsample_rate', 2)

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        data_args.model_type = "qwen3vl"

    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
        )
        data_args.image_processor = Qwen2VLImageProcessor.from_pretrained(
            model_args.model_name_or_path,
        )
        data_args.model_type = "qwen2vl"

    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    save_code_root = os.path.join(
        os.getcwd(), 
        os.path.basename(training_args.output_dir),
        "code_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    if torch.distributed.get_rank() == 0 and not os.path.exists(save_code_root):
        os.makedirs(save_code_root)
        os.system(f"cp -r {os.getcwd()}/src {save_code_root}/src")
        os.system(f"mkdir -p {save_code_root}/scripts")
        os.system(f"cp -r {os.getcwd()}/scripts/evaluation {save_code_root}/scripts/evaluation")
        os.system(f"cp -r {os.getcwd()}/scripts/train {save_code_root}/scripts/train")
        os.system(f"cp -r {os.getcwd()}/scripts/*.json {save_code_root}/scripts/train/")

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    except:
        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     model_args.model_name_or_path,
        #     model_max_length=training_args.model_max_length,
        #     padding_side="right",
        #     revision="main",  # 或者具体的commit hash
        #     trust_remote_code=True,
        # )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
        )


    if model_args.add_regressive_token:
        new_tokens = []
        for num in np.linspace(-2,4,601):
            num = int(num*100)/100
            new_tokens.append(str(num))
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
    

    exclude_geometry_encoder_on_save = getattr(model_args, "exclude_geometry_encoder_on_save", False)
    set_model(model_args, model)
    if exclude_geometry_encoder_on_save:
        setattr(
            model,
            "_exclude_geometry_encoder_on_save",
            bool(exclude_geometry_encoder_on_save),
        )

    if torch.distributed.get_rank() == 0:
        if hasattr(model.visual, 'print_trainable_parameters'):
            model.visual.print_trainable_parameters()
        if hasattr(model.model, 'print_trainable_parameters'):
            model.model.print_trainable_parameters()

    rank0_print(model.config)

    print_trainable_parameter_summary(model)

    rank0_print('====                                          ====')
    rank0_print('====  Only training the following parameters  ====')
    rank0_print('====                                          ====')
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            rank0_print('\t', name, param.shape)


    if model_args.use_geometry_encoder or model_args.force_simulate_vggt:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder or model_args.force_simulate_vggt)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    callbacks = [LossBreakdownCallback()]

    if exclude_geometry_encoder_on_save:
        trainer_cls = GeoAwareTrainer
    else:
        trainer_cls = Trainer

    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        **data_module,
    )

    report_to = training_args.report_to
    if isinstance(report_to, str):
        report_targets = [report_to]
    elif report_to is None:
        report_targets = []
    else:
        report_targets = list(report_to)

    if "tensorboard" in report_targets:
        trainer.remove_callback(TensorBoardCallback)
        trainer.add_callback(VisualTensorBoardCallback())

    force_resume_checkpoint = os.environ.get("FORCE_RESUME_CHECKPOINT", "").strip()
    resume_checkpoint_path = None
    checkpoint_candidates = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))

    if force_resume_checkpoint:
        if force_resume_checkpoint.lower() == "last":
            checkpoint_candidates = sorted(
                checkpoint_candidates,
                key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
            )
            if checkpoint_candidates:
                resume_checkpoint_path = str(checkpoint_candidates[-1])
        else:
            resume_checkpoint_path = force_resume_checkpoint

    if resume_checkpoint_path is None and checkpoint_candidates:
        checkpoint_candidates = sorted(
            checkpoint_candidates,
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1,
        )
        resume_checkpoint_path = str(checkpoint_candidates[-1])
        

    if resume_checkpoint_path is not None:
        rank0_print(f"[train_qwen] resume from checkpoint: {resume_checkpoint_path}")
        trainer.train(resume_from_checkpoint=resume_checkpoint_path)
    else:
        rank0_print("[train_qwen] no checkpoint found, start training from scratch")
        trainer.train()

    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        exclude_geometry_encoder_on_save=exclude_geometry_encoder_on_save,
    )


if __name__ == "__main__":

    train(attn_implementation="flash_attention_2")
