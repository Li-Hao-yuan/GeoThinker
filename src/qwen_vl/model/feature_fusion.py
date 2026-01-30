"""Feature fusion modules for combining 2D and 3D features."""

import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
import math
import numpy as np
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor

#####

@dataclass
class FeatureFusionConfig:
    """Configuration for feature fusion."""
    fusion_method: str = "add"  # "add", "concat", "gated", "weighted", "cross_attention", "zero"
    hidden_size: int = 3584
    num_heads: int = 8
    dropout: float = 0.1
    num_layers: int = 1

class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with position encoding, MLP and residual connections."""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer norms
        self.norm1_query = nn.LayerNorm(hidden_size)
        self.norm1_key = nn.LayerNorm(hidden_size)
        self.norm1_value = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
    def get_2d_sincos_pos_embed(self, height: int, width: int, embed_dim: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings.
        
        Args:
            height: Height of the grid
            width: Width of the grid  
            embed_dim: Embedding dimension
            device: Device to create tensor on
            
        Returns:
            pos_embed: Position embeddings of shape [height*width, embed_dim]
        """
        # Generate grid coordinates
        grid_h = torch.arange(height, dtype=torch.float32, device=device)
        grid_w = torch.arange(width, dtype=torch.float32, device=device)
        grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
        grid = torch.stack(grid, dim=0)  # [2, height, width]
        
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed
        
    def get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D sinusoidal position embeddings from grid.
        
        Args:
            embed_dim: Embedding dimension
            grid: Grid coordinates of shape [2, height, width]
            
        Returns:
            pos_embed: Position embeddings of shape [height*width, embed_dim]
        """
        assert embed_dim % 2 == 0
        
        # Use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # [height*width, embed_dim//2]
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # [height*width, embed_dim//2]
        
        emb = torch.cat([emb_h, emb_w], dim=1)  # [height*width, embed_dim]
        return emb
        
    def get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
        """
        Generate 1D sinusoidal position embeddings.
        
        Args:
            embed_dim: Embedding dimension
            pos: Position tensor of shape [height, width]
            
        Returns:
            emb: Position embeddings of shape [height*width, embed_dim]
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # [embed_dim//2]
        
        pos = pos.flatten()
        out = torch.einsum('m,d->md', pos, omega)  # [height*width, embed_dim//2], outer product
        
        emb_sin = torch.sin(out)  # [height*width, embed_dim//2]
        emb_cos = torch.cos(out)  # [height*width, embed_dim//2]
        
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # [height*width, embed_dim]
        return emb
        
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor, h_grid: int, w_grid: int) -> torch.Tensor:
        # Normalize features
        query = self.norm1_query(features_2d)
        key = self.norm1_key(features_3d)
        value = self.norm1_value(features_3d)
        
        # Add batch dimension if needed
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Generate 2D position embeddings
        pos_embed = self.get_2d_sincos_pos_embed(h_grid, w_grid, self.hidden_size, query.device).to(query.dtype)  # [h_grid*w_grid, hidden_size]

        # Add position embeddings to query and key
        # Assuming features are organized as [batch_size, h_grid*w_grid, hidden_size]

        query = query + pos_embed.unsqueeze(0)  # Broadcast across batch dimension
        key = key + pos_embed.unsqueeze(0)
            
        # Cross-attention: 2D features as query, 3D features as key/value
        attn_output, _ = self.cross_attention(query, key, value)
        
        if squeeze_output:
            attn_output = attn_output.squeeze(0)
            
        # First residual connection
        x = features_2d + attn_output
        
        # MLP with second residual connection
        mlp_output = self.mlp(self.norm2(x))
        x = x + mlp_output
        
        return x

class FeatureFusionModule(nn.Module):
    """Enhanced feature fusion module with multiple fusion strategies."""
    
    def __init__(self, config: FeatureFusionConfig):
        super().__init__()
        self.config = config
        self.fusion_method = config.fusion_method
        self.hidden_size = config.hidden_size
        
        self._build_fusion_layers()
    
    def _build_fusion_layers(self):
        """Build fusion layers based on method."""
        if self.config.fusion_method == "concat":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.projection = nn.Linear(self.hidden_size * 2, self.hidden_size)
            
        elif self.config.fusion_method == "cross_attention":
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_size, 
                    self.config.num_heads, 
                    self.config.dropout
                ) 
                for _ in range(self.config.num_layers)
            ])

        elif self.config.fusion_method == "gated":
            self.norm1 = nn.LayerNorm(self.hidden_size)
            self.norm2 = nn.LayerNorm(self.hidden_size)
            self.gate_projection = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size),
                nn.Sigmoid()
            )
            
        elif self.config.fusion_method == "weighted":
            self.weight_2d = nn.Parameter(torch.tensor(0.))
            self.weight_3d = nn.Parameter(torch.tensor(1.))
    
    def forward(self, features_2d: torch.Tensor, features_3d: torch.Tensor) -> torch.Tensor:
        """
        Fuse 2D and 3D features.
        
        Args:
            features_2d: 2D image features
            features_3d: 3D geometry features
        Returns:
            Fused features
        """

        batch, h_grid, w_grid, hidden_dim = features_3d.shape

        if self.fusion_method == "add":
            fusion_feature =  features_2d + features_3d
            
        elif self.fusion_method == "concat":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            fusion_feature = self.projection(concat_features)
            
        elif self.fusion_method == "cross_attention":
            features_2d = features_2d.view(features_2d.size(0), -1, self.hidden_size)  # Flatten spatial dimensions
            features_3d = features_3d.view(features_3d.size(0), -1, self.hidden_size)
            x = features_2d
            for block in self.cross_attn_blocks:
                x = block(x, features_3d, h_grid, w_grid)
            fusion_feature = x
            
        elif self.fusion_method == "gated":
            features_2d = self.norm1(features_2d)
            features_3d = self.norm2(features_3d)
            concat_features = torch.cat([features_2d, features_3d], dim=-1)
            gate = self.gate_projection(concat_features)
            fusion_feature = gate * features_2d + (1 - gate) * features_3d
            
        elif self.fusion_method == "weighted":
            # Normalize weights to sum to 1
            weight_sum = self.weight_2d + self.weight_3d
            norm_weight_2d = self.weight_2d / weight_sum
            norm_weight_3d = self.weight_3d / weight_sum
            fusion_feature = norm_weight_2d * features_2d + norm_weight_3d * features_3d
        elif self.fusion_method == "zero":
            fusion_feature = features_2d #+ features_3d * 0
        elif self.fusion_method == "only_3d":
            fusion_feature =  features_3d
        elif self.fusion_method == "none":
            fusion_feature =  features_2d - features_2d

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    
        return fusion_feature, features_3d

class GeometryFeatureMerger(nn.Module):
    """Unified merger for geometry features from different encoders.
    
    Supports different merger types:
    - "mlp": MLP-based feature transformation with spatial merging
    - "avg": Average pooling across spatial merge dimensions
    - "attention": Attention-based merger (not implemented yet)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, context_dim: int, 
                 spatial_merge_size: int = 2, merger_type: str = "mlp", 
                 depart_smi_token = False, smi_image_num = 8, smi_downsample_rate = 2):
        super().__init__()
        self.merger_type = merger_type
        self.input_dim = context_dim * (spatial_merge_size ** 2)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.merge_size = spatial_merge_size
        self.depart_smi_token = depart_smi_token
        self.smi_image_num = smi_image_num
        self.smi_downsample_rate = smi_downsample_rate
        
        if merger_type == "mlp":
            # Import here to avoid circular import
            try:
                from .modeling_qwen2_5_vl import Qwen2RMSNorm
            except ImportError:
                # Fallback to standard LayerNorm if Qwen2RMSNorm not available
                Qwen2RMSNorm = nn.LayerNorm
                
            self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
            self.mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "avg":
            self.mlp = nn.Sequential(
                nn.Linear(context_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )
        elif merger_type == "attention":
            # Add attention-based merger for future extensibility
            raise NotImplementedError("Attention merger not implemented yet")
        else:
            raise ValueError(f"Unknown merger type: {merger_type}")
        
    
    def forward(self, x: torch.Tensor):
        """Forward pass through the merger."""

        # torch.Size([1, 28, 37, 2048])
        # torch.Size([8, 28, 37, 2048])

        n_image, h_patch, w_patch, dim = x.shape

        if self.depart_smi_token and n_image > self.smi_image_num:
            x = x.permute(0, 3, 1, 2)
            x = torch.nn.functional.interpolate(
                x, 
                size=(h_patch//self.smi_downsample_rate, w_patch//self.smi_downsample_rate), 
                mode="bilinear", 
                align_corners=False
            )
            x = x.permute(0, 2, 3, 1)

            n_image, h_patch, w_patch, dim = x.shape

        x = x[:, :h_patch // self.merge_size * self.merge_size, :w_patch // self.merge_size*self.merge_size , :]
        x = x.reshape(n_image, h_patch // self.merge_size, self.merge_size, w_patch // self.merge_size, self.merge_size, dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        if self.merger_type == "mlp":
            x = self.mlp(self.ln_q(x).view(-1, self.input_dim))
        elif self.merger_type == "avg":
            # Average pooling across spatial merge dimensions
            x = x.mean(dim=(3, 4))  # Average over the merge_size dimensions
            x = x.view(-1, dim)  # Flatten for projection
            x = self.mlp(x)
        else:
            raise NotImplementedError(f"Merger type {self.merger_type} not implemented")
        x = x.reshape(n_image, h_patch // self.merge_size, w_patch // self.merge_size, -1)

        return x
