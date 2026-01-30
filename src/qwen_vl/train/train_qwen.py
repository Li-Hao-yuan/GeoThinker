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
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent.parent
# print("project_root",project_root)
sys.path.insert(0, str(project_root))

import qwen_vl.train.trainer
import qwen_vl.train.sampler
from trainer import replace_qwen2_vl_attention_class

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

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False



    if model_args.tune_mm_llm:
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.named_parameters():
            if "cross_attn" in n: p.requires_grad = True
            else: p.requires_grad = False
        model.lm_head.requires_grad = False


    if model_args.use_geometry_encoder:
        # vggt is frozen
        for n, p in model.geometry_encoder.named_parameters():
            p.requires_grad = False

def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)
    # enable_full_determinism(training_args.seed)

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    if "qwen2.5" in model_args.model_name_or_path.lower() or "OUTPUT" in model_args.model_name_or_path:
        if not model_args.use_geometry_encoder:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        else:
            from qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT
            print(model_args.model_name_or_path)
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            setattr(config, "geo_cross_attn", model_args.geo_cross_attn)
            setattr(config, "geo_inject_version", model_args.geo_inject_version)
            setattr(config, "geo_importance_gate", model_args.geo_importance_gate)
            setattr(config, "use_qwenvl_loss", model_args.use_qwenvl_loss)
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame",
                "feature_fusion_method", 
                "fusion_num_layers",
                "geometry_merger_type",
                "depart_smi_token",
                "smi_image_num",
                "smi_downsample_rate",
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = Qwen2_5_VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                geometry_encoder_path=model_args.geometry_encoder_path
            )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        if getattr(config, 'depart_smi_token', False):
            data_args.depart_smi_token = True
            data_args.smi_image_num = getattr(config, 'smi_image_num', 8)
            data_args.smi_downsample_rate = getattr(config, 'smi_downsample_rate', 2)

        data_args.model_type = "qwen2.5vl"
    elif "qwen3" in model_args.model_name_or_path.lower():
        if not model_args.use_geometry_encoder:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
            )
        else:
            from qwen_vl.model.qwenvl3.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT
            print(model_args.model_name_or_path)
            config = AutoConfig.from_pretrained(model_args.model_name_or_path)
            setattr(config.text_config, "depart_smi_token", model_args.depart_smi_token)
            setattr(config.text_config, "smi_image_num", model_args.smi_image_num)
            setattr(config.text_config, "smi_downsample_rate", model_args.smi_downsample_rate)

            setattr(config.text_config, "geo_cross_attn", model_args.geo_cross_attn)
            setattr(config.text_config, "geo_inject_version", model_args.geo_inject_version)
            setattr(config.text_config, "geo_importance_gate", model_args.geo_importance_gate)
            if hasattr(config, "use_geometry_encoder") and config.use_geometry_encoder != model_args.use_geometry_encoder:
                raise ValueError(
                    "The use_geometry_encoder in config and model_args are not consistent. "
                    "Please check the model config."
                )

            for k in [
                "use_geometry_encoder", 
                "geometry_encoder_type", 
                "reference_frame",
                "feature_fusion_method",  
                "fusion_num_layers",
                "geometry_merger_type",
                "depart_smi_token",
                "smi_image_num",
                "smi_downsample_rate",
            ]:
                setattr(config, k, getattr(model_args, k))

            assert model_args.geometry_encoder_path is not None, \
                "geometry_encoder_path must be set in the config when use_geometry_encoder is True."
            model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
                pretrained_model_name_or_path=model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                geometry_encoder_path=model_args.geometry_encoder_path
            )

        data_args.image_processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
        ).image_processor

        if getattr(config, 'depart_smi_token', False):
            data_args.depart_smi_token = True
            data_args.smi_image_num = getattr(config, 'smi_image_num', 8)
            data_args.smi_downsample_rate = getattr(config, 'smi_downsample_rate', 2)

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

    set_model(model_args, model)

    if torch.distributed.get_rank() == 0:
        model.visual.print_trainable_parameters()
        model.model.print_trainable_parameters()

    rank0_print(model.config)

    rank0_print('====                                          ====')
    rank0_print('====  Only training the following parameters  ====')
    rank0_print('====                                          ====')
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            rank0_print('\t', name, param.shape)

    if model_args.use_geometry_encoder:
        setattr(data_args, "use_geometry_encoder", model_args.use_geometry_encoder)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    data_args.image_processor.save_pretrained(training_args.output_dir)

    source_path = os.path.join(model_args.model_name_or_path, "chat_template.json")
    template_path = os.path.join(training_args.output_dir, "chat_template.json")
    shutil.copy2(source_path, template_path)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":

    train(attn_implementation="flash_attention_2")
