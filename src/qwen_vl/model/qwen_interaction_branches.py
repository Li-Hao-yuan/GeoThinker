"""Final GeoThinker-F interaction module over GeoThinker.

These research interaction implementations are covered by the repository MIT
License. Upstream Qwen-derived model files retain their separate notices; see
the repository NOTICE.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen_interaction import *  # noqa: F401,F403


class QwenVGGTInteractionv3(nn.Module):
    """
    GeoThinker-F hybrid interaction: even-indexed layers use texture features,
    and odd-indexed layers use predicted temporal hidden states.
    """

    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        total_layers = getattr(config, "num_hidden_layers", 1)
        self.texture_layer_flag = layer_idx % 2 == 0

        if hasattr(config, "vision_config"):
            self.pooling_stride = config.vision_config.spatial_merge_size
        else:
            self.pooling_stride = 2

        depart_smi_token = kwargs.pop("depart_smi_token", False)
        if depart_smi_token:
            self.pooling_stride *= kwargs.pop("smi_downsample_rate", 2)

        feature_dim = int(getattr(config, "geometry_encoder_feature_dim", config.hidden_size))

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vggt_layernorm = nn.LayerNorm(feature_dim, eps=config.rms_norm_eps)
        self.temporal_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        if self.texture_layer_flag:
            self.i_proj = nn.Linear(feature_dim, config.hidden_size, bias=False)
        else: self.i_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        self.gate = nn.Parameter(torch.tensor(0.0))

    def _reshape_by_grid(self, semantic_hidden, source_hidden, grid_thw):
        B_orig, S_total, D = semantic_hidden.shape
        q_input = semantic_hidden
        k_input = source_hidden
        v_input = source_hidden

        is_reshaped = False
        h_feat, w_feat = 0, 0
        B, S = B_orig, S_total

        if grid_thw is not None:
            H_orig, W_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
            h_feat = H_orig // self.pooling_stride
            w_feat = W_orig // self.pooling_stride
            tokens_per_img = h_feat * w_feat

            if tokens_per_img > 0 and S_total % tokens_per_img == 0:
                num_images = S_total // tokens_per_img
                q_input = semantic_hidden.view(num_images, tokens_per_img, D)
                k_input = source_hidden.view(num_images, tokens_per_img, D)
                v_input = source_hidden.view(num_images, tokens_per_img, D)
                is_reshaped = True
                B, S = num_images, tokens_per_img

        return q_input, k_input, v_input, B, S, B_orig, S_total, D, is_reshaped, h_feat, w_feat

    def _forward_texture_v6(self, q_input, k_input, v_input, B, S, D, is_reshaped, B_orig, S_total, h_feat, w_feat):
        q = self.q_proj(q_input).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(k_input).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(v_input).view(B, S, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        importance = self.importance_net(q_input)
        importance_logit = torch.log(importance + 0.1).view(B, 1, 1, S)
        attn_weights = attn_weights + importance_logit

        attn_probs = F.softmax(attn_weights, dim=-1)
        output = attn_probs @ v
        output = output.transpose(1, 2).reshape(B, S, D)

        if is_reshaped:
            output = output.view(B_orig, S_total, D)

        output = self.o_proj(output)
        return torch.tanh(self.gate) * output

    def _forward_temporal_v41(self, q_input, k_input, v_input, B, S, D, is_reshaped, B_orig, S_total):
        q = self.q_proj(q_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_weights, dim=-1)
        output = attn_probs @ v
        output = output.transpose(1, 2).reshape(B, S, D)

        if is_reshaped:
            output = output.view(B_orig, S_total, self.hidden_size)

        output = self.o_proj(output)
        return torch.tanh(self.gate) * output

    def forward(self, semantic_hidden, vggt_features=None, next_hidden_states=None, grid_thw=None, **kwargs):
        use_texture_branch = self.layer_idx is not None and self.texture_layer_flag

        semantic_hidden = self.input_layernorm(semantic_hidden)

        if use_texture_branch:
            if vggt_features is None:
                raise ValueError("QwenVGGTInteractionv3 texture branch requires vggt_features.")

            source_hidden = self.vggt_layernorm(vggt_features)
            source_hidden = self.i_proj(source_hidden)
            (
                q_input,
                k_input,
                v_input,
                B,
                S,
                B_orig,
                S_total,
                D,
                is_reshaped,
                h_feat,
                w_feat,
            ) = self._reshape_by_grid(semantic_hidden, source_hidden, grid_thw)

            return self._forward_texture_v6(
                q_input,
                k_input,
                v_input,
                B,
                S,
                D,
                is_reshaped,
                B_orig,
                S_total,
                h_feat,
                w_feat,
            )

        if next_hidden_states is None:
            raise ValueError("QwenVGGTInteractionv3 temporal branch requires next_hidden_states.")

        source_hidden = self.temporal_layernorm(next_hidden_states)
        source_hidden = self.i_proj(source_hidden)

        (
            q_input,
            k_input,
            v_input,
            B,
            S,
            B_orig,
            S_total,
            D,
            is_reshaped,
            _,
            _,
        ) = self._reshape_by_grid(semantic_hidden, source_hidden, grid_thw)

        return self._forward_temporal_v41(
            q_input,
            k_input,
            v_input,
            B,
            S,
            D,
            is_reshaped,
            B_orig,
            S_total,
        )

