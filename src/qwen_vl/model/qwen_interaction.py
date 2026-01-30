import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func, flash_attn_func
    from flash_attn.layers.rotary import apply_rotary_emb

else:
    flash_attn_varlen_func = None
    apply_rotary_emb = None



class QwenVGGTInteractionv1(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        vggt_dim = hidden_size
        num_heads = config.num_attention_heads

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # 1. Query Projector (来自 LLM Hidden State)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vggt_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # 2. Key/Value Projector (来自 VGGT)
        # 注意：VGGT维度可能和LLM不一样，这里做对齐
        self.k_proj = nn.Linear(vggt_dim, hidden_size, bias=False)
        self.v_proj = nn.Linear(vggt_dim, hidden_size, bias=False)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0)) # Tanh Gate
        self.layer_idx = layer_idx

    def forward(self, image_hidden_states, vggt_features):
        # hidden_states: [Batch, Seq_Len_Text, Hidden]
        # vggt_features: [Batch, Seq_Len_Image, VGGT_Dim]

        # --- Projection ---
        image_hidden_states = self.input_layernorm(image_hidden_states)
        vggt_features = self.vggt_layernorm(vggt_features)

        q = self.q_proj(image_hidden_states)
        k = self.k_proj(vggt_features)
        v = self.v_proj(vggt_features)

        # Reshape for Flash Attn: [Batch, Seq, Heads, Dim]
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim)
        
        # --- Flash Attention (Cross) ---
        # Flash Attn 自动处理 Q 和 K/V 长度不一致的情况
        # causal=False (Cross Attention 不需要 causal mask)
        attn_output = flash_attn_func(q, k, v, causal=False)
        
        # Reshape back
        attn_output = attn_output.reshape(image_hidden_states.shape)
        
        # --- Output Projection & Gating ---
        attn_output = self.o_proj(attn_output)
        attn_output = torch.tanh(self.gate) * attn_output

        return attn_output


class QwenVGGTInteractionv2(nn.Module):
    def __init__(self, config, layer_idx=None, use_spatial_bias=True, use_importance_gate=True, geo_learn_bias=True, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        self.use_spatial_bias = use_spatial_bias
        self.learnable_bias = geo_learn_bias
        self.use_importance_gate = use_importance_gate
        
        # 假设 pooling stride = 2 (28->14, 36->18)
        if hasattr(config, "vision_config"):
            self.pooling_stride = config.vision_config.spatial_merge_size
        else: self.pooling_stride = 2

        depart_smi_token = kwargs.pop("depart_smi_token", False)
        if depart_smi_token: self.pooling_stride *= kwargs.pop("smi_downsample_rate", 2)

        # 1. Norm
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vggt_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 2. Projectors
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # 3. [理念3] 背景抑制
        if self.use_importance_gate:
            self.importance_net = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 4, 1), 
                nn.Sigmoid()
            )

        self.gate = nn.Parameter(torch.tensor(0.0))

        # 5. [理念2] 空间偏置系数
        self.bias_gate = nn.Sequential(
            nn.Linear(config.hidden_size, config.num_attention_heads // 2), # 为每个 head 生成独立的 gate
            nn.Sigmoid()
        )

    def get_spatial_bias(self, h_feat, w_feat, device, dtype):
        """
        生成单张特征图的 Bias 矩阵: [N, N], 其中 N = h_feat * w_feat
        """
        y = torch.arange(h_feat, device=device, dtype=dtype)
        x = torch.arange(w_feat, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # [N, 2]
        coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)
        
        # [N, N]
        dist_matrix = torch.cdist(coords, coords, p=2)
        
        # 归一化: 除以对角线长度，让距离在 0~1 之间比较合理
        diag_len = math.sqrt(h_feat**2 + w_feat**2) + 1e-6
        dist_matrix = dist_matrix / diag_len
        
        # Bias = -dist * scale
        return -dist_matrix 

    def forward(self, semantic_hidden, vggt_features, grid_thw, **kwargs):
        """
        semantic_hidden: [B(1), Total_Seq(8064), Dim]
        vggt_features:   [B(1), Total_Seq(8064), Dim]
        grid_thw:        [Batch, 3] -> 这里的 H, W 是原始图片大小 (28, 36)
        """
        # Norm First
        semantic_hidden = self.input_layernorm(semantic_hidden)
        vggt_features = self.vggt_layernorm(vggt_features)
        
        # 获取原始 Batch 和 Total_Seq
        B_orig, S_total, D = semantic_hidden.shape
        
        # === 核心修改：处理多图 Batching ===
        # 默认按照 sequence 处理 (fallback)
        q_input = semantic_hidden
        k_input = vggt_features
        v_input = vggt_features
        
        # 记录是否进行了 reshape，以便最后还原
        is_reshaped = False 
        h_feat, w_feat = 0, 0

        ## spatial bias here
        if grid_thw is not None:
            # 获取原始 H, W (28, 36)
            H_orig, W_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
            
            # 计算 Pooling 后的特征图尺寸 (14, 18)
            h_feat = H_orig // self.pooling_stride
            w_feat = W_orig // self.pooling_stride
            tokens_per_img = h_feat * w_feat
            
            # 检查能否整除，确保对齐
            if S_total % tokens_per_img == 0:
                num_images = S_total // tokens_per_img
                
                # [关键操作] 重塑维度:
                # [1, 8064, D] -> [32, 252, D]
                # 这样每张图只和自己做 Attention，这也是一种极致的“Locality”
                q_input = semantic_hidden.view(num_images, tokens_per_img, D)
                k_input = vggt_features.view(num_images, tokens_per_img, D)
                v_input = vggt_features.view(num_images, tokens_per_img, D)
                is_reshaped = True
                
                # 更新当前的 Batch 和 Seq
                B, S = num_images, tokens_per_img
            else:
                # 如果对不齐（可能有padding token），则退回原始维度
                B, S = B_orig, S_total
        else:
            B, S = B_orig, S_total

        # === 投影 ===
        q = self.q_proj(q_input).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(k_input).view(B, S, self.num_heads, self.head_dim)
        v = self.v_proj(v_input).view(B, S, self.num_heads, self.head_dim)

        # Transpose: [B, Heads, S, S]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention Score
        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # === [理念2] 空间偏置 (现在只需计算单张图的) ===
        if self.use_spatial_bias and is_reshaped:
            spatial_bias = self.get_spatial_bias(h_feat, w_feat, q.device, q.dtype)

            gate_score = self.bias_gate(q_input)
            gate_score = gate_score.transpose(1, 2).unsqueeze(-1)

            n_heads = self.num_heads
            split_idx = n_heads // 2

            attn_weights[:, :split_idx, :, :] += spatial_bias * gate_score

        if self.use_importance_gate:
            importance = self.importance_net(q_input) # 使用 vggt feature 判断

            # importance_logit = torch.log(importance + 1e-6)
            importance_logit = torch.log(importance + 0.1)

            importance_logit = importance_logit.view(B, 1, 1, S)
            attn_weights = attn_weights + importance_logit


        attn_probs = F.softmax(attn_weights, dim=-1)

        output = attn_probs @ v # [B, Heads, S, Head_Dim]
        
        # === 还原维度 ===
        output = output.transpose(1, 2).reshape(B, S, D)
        
        if is_reshaped:
            # [32, 252, D] -> [1, 8064, D]
            output = output.view(B_orig, S_total, D)

        output = self.o_proj(output)
        return torch.tanh(self.gate) * output


class QwenVGGTInteractionv2Flash(QwenVGGTInteractionv2):

    def forward(self, semantic_hidden, vggt_features, grid_thw, **kwargs):
        # Norm
        semantic_hidden = self.input_layernorm(semantic_hidden)
        vggt_features = self.vggt_layernorm(vggt_features)
        
        B_orig, S_total, D = semantic_hidden.shape
        
        # === Batching Strategy ===
        q_input, k_input, v_input = semantic_hidden, vggt_features, vggt_features
        is_reshaped = False 
        h_feat, w_feat = 0, 0

        # ... (保持原本的 Reshape 逻辑) ...
        if grid_thw is not None:
            H_orig, W_orig = grid_thw[0, 1].item(), grid_thw[0, 2].item()
            h_feat = H_orig // self.pooling_stride
            w_feat = W_orig // self.pooling_stride
            tokens_per_img = h_feat * w_feat
            
            if tokens_per_img > 0 and S_total % tokens_per_img == 0:
                num_images = S_total // tokens_per_img
                q_input = semantic_hidden.view(num_images, tokens_per_img, D)
                k_input = vggt_features.view(num_images, tokens_per_img, D)
                v_input = vggt_features.view(num_images, tokens_per_img, D)
                is_reshaped = True
                B, S = num_images, tokens_per_img
            else:
                B, S = B_orig, S_total
        else:
            B, S = B_orig, S_total

        # === Projections ===
        # SDPA 期望的输入维度是 [Batch, Heads, Seq, Head_Dim]
        q = self.q_proj(q_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v_input).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # === 核心修改：构建 Attention Bias ===
        attn_bias = None

        if self.use_spatial_bias and is_reshaped:
            # gate_score: [B, S, Heads] -> [B, Heads, S, 1] 为了广播
            # import pdb
            # pdb.set_trace()

            gate_score = self.bias_gate(k_input).transpose(1, 2).unsqueeze(-1)
            
            # spatial_bias: [S, S]
            spatial_bias = self.get_spatial_bias(h_feat, w_feat, q.device, q.dtype)

            head_specific_bias = torch.zeros(B, self.num_heads, S, S, 
                                           device=q.device, dtype=q.dtype)

            n_heads = self.num_heads
            split_idx = n_heads // 2
            
            # Combine: [B, Heads, S, S]
            # 注意：这里会产生显存开销，构建了一个完整的 Attention Map 大小的 Bias
            head_specific_bias[:, :split_idx, :, :] += spatial_bias * gate_score
            
            attn_bias = head_specific_bias

        # 2. Importance Gate Bias
        if self.use_importance_gate:
            importance = self.importance_net(q_input)
            importance_logit = torch.log(importance + 0.1) # [B, S, 1]
            
            mask_term = importance_logit.view(B, 1, 1, S)
            
            if attn_bias is None:
                attn_bias = mask_term
            else:
                attn_bias = attn_bias + mask_term

            # import pdb
            # pdb.set_trace()

            ### visual
            if False:
                import os, time

                # [3, 3, 392, 518]
                # [1, 576, 2560]
                # [ 1, 24, 32]

                save_root = "/home/ma-user/work/l30081110/VGLLM/visual/qwen3vl/gate/vsibench"
                os.makedirs(save_root, exist_ok=True)

                folder_names = os.listdir(save_root)
                folder_names.sort(key=lambda x: int(x))

                if len(folder_names) == 0:
                    os.makedirs(os.path.join(save_root, "1"))
                    save_path = os.path.join(save_root, "1", str(self.layer_idx)+".pth")
                elif os.path.exists(os.path.join(save_root, folder_names[-1], str(self.layer_idx)+".pth")):
                    target_folder_name = str( int(folder_names[-1]) + 1 )
                    save_path = os.path.join(save_root, target_folder_name, str(self.layer_idx)+".pth")
                    os.makedirs("/".join(save_path.split("/")[:-1]), exist_ok=True)
                else:
                    save_path = os.path.join(save_root, folder_names[-1], str(self.layer_idx)+".pth")
                
                torch.save({
                    "importance": importance.cpu(),
                    "grid_thw": grid_thw.cpu(),
                }, save_path)

                # time.sleep(1)

        # === Flash Attention (SDPA) ===
        # PyTorch 2.0+ 会自动选择最优 kernel (FlashAttn, MemEfficient, or Math)
        # 传入 attn_mask 即等同于在 Softmax 之前加上这个 Tensor
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)

        # === Output Projection ===
        output = output.transpose(1, 2).contiguous().reshape(B, S, self.hidden_size)
        
        if is_reshaped:
            output = output.view(B_orig, S_total, self.hidden_size)

        output = self.o_proj(output)
        return torch.tanh(self.gate) * output


