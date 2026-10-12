from __future__ import annotations
import math
from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    Splits [x1, x2] -> [-x2, x1]
    
    Args:
        x: [..., D] where D is even
    Returns:
        [..., D] rotated tensor
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    freqs: torch.Tensor, 
    t: torch.Tensor, 
    start_index: int = 0, 
    scale: float = 1.0
) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.
    
    Args:
        freqs: [1, 1, L, rot_dim] or [L, rot_dim] frequency tensor
        t: [B, H, L, D] input tensor (query or key)
        start_index: starting index in D dimension to apply RoPE
        scale: scaling factor for rotary embeddings
    
    Returns:
        [B, H, L, D] tensor with RoPE applied
    """
    freqs = freqs.to(t.device, t.dtype)
    if freqs.ndim == 2:
        freqs = freqs.unsqueeze(0).unsqueeze(0)
    
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    
    assert rot_dim <= t.shape[-1], f"feature dim {t.shape[-1]} < rope dim {rot_dim}"
    assert rot_dim % 2 == 0, f"rot_dim must be even, got {rot_dim}"
    
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    t_rot = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)
    
    return torch.cat((t_left, t_rot, t_right), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) with optional learning and caching.
    
    Reference: https://arxiv.org/abs/2104.09864
    """
    
    def __init__(
        self, 
        dim: int, 
        theta: float = 10000.0, 
        learned_freq: bool = False, 
        interpolate_factor: float = 1.0
    ):
        """
        Args:
            dim: Embedding dimension (must be even)
            theta: Base for frequency computation (typically 10000)
            learned_freq: If True, frequencies are learnable parameters
            interpolate_factor: Factor for position interpolation (>= 1.0)
        """
        super().__init__()
        
        assert dim % 2 == 0, f"dim must be even for RoPE, got {dim}"
        assert interpolate_factor >= 1.0, f"interpolate_factor must be >= 1.0, got {interpolate_factor}"
        assert theta > 0, f"theta must be positive, got {theta}"
        
        self.dim = dim
        self.theta = theta
        self.learned_freq = learned_freq
        self.interpolate_factor = interpolate_factor
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        if learned_freq:
            self.freqs = nn.Parameter(inv_freq)
        else:
            self.register_buffer('freqs', inv_freq, persistent=True)
        
        self.cache: Dict[Tuple[str, int, str, str], torch.Tensor] = {}
    
    def _compute_freqs(
        self, 
        L: int, 
        device: torch.device, 
        dtype: torch.dtype, 
        offset: float = 0.0
    ) -> torch.Tensor:
        """
        Compute frequency tensor for sequence length L.
        
        Args:
            L: Sequence length
            device: Target device
            dtype: Target dtype
            offset: Position offset (default 0)
        
        Returns:
            [L, dim] frequency tensor where:
                - First dim//2 columns: same frequencies
                - Last dim//2 columns: same frequencies (duplicated for complex pairs)
        """
        pos = torch.arange(L, device=device, dtype=torch.float32) + offset
        
        pos = pos / self.interpolate_factor
        
        freqs_half = torch.outer(pos, self.freqs.to(device).to(torch.float32))
        
        freqs = torch.cat([freqs_half, freqs_half], dim=-1)
        
        return freqs.to(dtype=dtype)
    
    def prepare_freqs_1d(
        self, 
        L: int, 
        device: torch.device, 
        dtype: torch.dtype, 
        offset: float = 0.0
    ) -> torch.Tensor:
        """
        Prepare 1D frequency tensor with caching (for non-learned) or fresh computation (for learned).
        
        Args:
            L: Sequence length
            device: Target device
            dtype: Target dtype
            offset: Position offset
        
        Returns:
            [L, dim] frequency tensor
        """
        if self.learned_freq:
            return self._compute_freqs(L, device, dtype, offset)
        
        cache_key = ("freqs1d", L, str(device), str(dtype))
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if cached.device == device:
                return cached
            else:
                cached = cached.to(device)
                self.cache[cache_key] = cached
                return cached
        
        with torch.no_grad():
            freqs = self._compute_freqs(L, device, dtype, offset)
        
        self.cache[cache_key] = freqs
        return freqs
    
    def clear_cache(self):
        """Clear the frequency cache."""
        self.cache.clear()
    
    def forward(
        self, 
        L: int, 
        device: torch.device = None, 
        dtype: torch.dtype = None,
        offset: float = 0.0
    ) -> torch.Tensor:
        """
        Forward pass (alias for prepare_freqs_1d).
        
        Args:
            L: Sequence length
            device: Target device (default: self.freqs.device)
            dtype: Target dtype (default: self.freqs.dtype)
            offset: Position offset
        
        Returns:
            [L, dim] frequency tensor
        """
        if device is None:
            device = self.freqs.device
        if dtype is None:
            dtype = self.freqs.dtype
        
        return self.prepare_freqs_1d(L, device, dtype, offset)
    
# ============================================================================
# Attention Components
# ============================================================================

class FullAttention(nn.Module):
    """Full attention with handling for fully masked rows and optional RoPE."""
    
    def __init__(
        self, 
        attention_dropout: float = 0.1, 
        output_attention: bool = False,
        use_rope: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.use_rope = use_rope

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None
    ):
        B, H, L, E = queries.shape
        _, _, S, _ = keys.shape
        scale = 1.0 / math.sqrt(E)
        
        # Apply RoPE if enabled
        if self.use_rope and freqs is not None:
            queries = apply_rotary_emb(freqs, queries)
            keys = apply_rotary_emb(freqs, keys)
        
        # Compute attention scores
        scores = torch.einsum("bhle,bhse->bhls", queries, keys) * scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        fully_masked = torch.all(scores == float('-inf'), dim=-1, keepdim=True)
        
        # Softmax and handle NaN
        A = torch.softmax(scores, dim=-1)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = A.masked_fill(fully_masked, 0.0)
        A = self.dropout(A)
        
        V = torch.einsum("bhls,bhse->bhle", A, values)
        V = V.masked_fill(fully_masked, 0.0)
        
        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None

class AttentionLayer(nn.Module):
    """Multi-head attention layer wrapper."""
    
    def __init__(
        self, 
        attention: nn.Module,
        d_model: int, 
        n_heads: int, 
        d_keys: Optional[int] = None, 
        d_values: Optional[int] = None
    ):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(
        self, 
        queries: torch.Tensor, 
        keys: torch.Tensor, 
        values: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        freqs: Optional[torch.Tensor] = None
    ):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        # Project and reshape
        queries = self.query_projection(queries).view(B, L, H, -1).transpose(1, 2)
        keys = self.key_projection(keys).view(B, S, H, -1).transpose(1, 2)
        values = self.value_projection(values).view(B, S, H, -1).transpose(1, 2)
        
        # Attention
        out, attn = self.inner_attention(
            queries, keys, values, attn_mask=attn_mask, freqs=freqs
        )
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.out_projection(out), attn

class MAGETimeBlock(nn.Module):
    def __init__(
        self, 
        num_granularities: int,
        d_model: int, 
        n_heads: int, 
        dropout: float = 0.1, 
        output_attention: bool = False, 
        no_inter_attn: bool = False, 
        use_rope_intra: bool = True,
        rope_dim_frac: float = 1.0, 
        rope_theta: float = 10000
    ):
        super().__init__()
        self.num_granularities = num_granularities
        self.d_model = d_model
        
        head_dim = d_model // n_heads
        rot_dim = max(2, int(head_dim * rope_dim_frac) // 2 * 2)
        
        self.intra_attentions = nn.ModuleList([
            AttentionLayer(
                FullAttention(
                    attention_dropout=dropout, 
                    output_attention=output_attention, 
                    use_rope=use_rope_intra
                ),
                d_model, n_heads
            ) for _ in range(num_granularities)
        ])
        
        self.intra_pre_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_granularities)
        ])
        
        self.intra_post_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_granularities)
        ])
        
        if no_inter_attn or num_granularities <= 1:
            self.inter_attention = None
            self.inter_pre_norm = None
            self.inter_post_norm = None
            self.inter_gates = None
            self.inter_output_norms = None
        else:
            self.inter_attention = AttentionLayer(
                FullAttention(
                    attention_dropout=dropout, 
                    output_attention=output_attention, 
                    use_rope=False
                ),
                d_model, n_heads
            )
            self.inter_pre_norm = nn.LayerNorm(d_model)
            self.inter_post_norm = nn.LayerNorm(d_model)
            self.inter_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model * 2, d_model),
                    nn.LayerNorm(d_model),
                    nn.Sigmoid()
                ) for _ in range(num_granularities)
            ])
            self.inter_output_norms = nn.ModuleList([
                nn.LayerNorm(d_model) for _ in range(num_granularities)
            ])
        
        self.rope_intra = RotaryEmbedding(dim=rot_dim, theta=rope_theta) if use_rope_intra else None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x_list: List[torch.Tensor], 
        mask_list: Optional[List[torch.Tensor]] = None
    ):
        """
        Args:
            x_list: List of [B, C, N_i, D] tensors
            mask_list: List of [B, C, N_i] or [C, N_i] where 1 = masked, 0 = non-masked for each scale
        Returns:
            x_out: List of [B, C, N_i, D] tensors
            attn_dict: Dictionary with 'intra' and optionally 'inter' attention weights
        """
        if mask_list is None:
            mask_list = [None] * len(x_list)
        
        x_intra = []
        attn_intra = []
        
        for i, (x_in, attn_layer, pre_norm, post_norm, mask_i) in enumerate(
            zip(x_list, self.intra_attentions, self.intra_pre_norms, self.intra_post_norms, mask_list)
        ):
            B, C, N_i, D = x_in.shape
            x_in_reshaped = x_in.view(B * C, N_i, D)
            attn_mask = None
            if mask_i is not None:
                if mask_i.ndim == 2:
                    assert mask_i.shape == (C, N_i), f"Mask shape {mask_i.shape} does not match expected [C={C}, N_i={N_i}]"
                    mask_i = mask_i.unsqueeze(0).expand(B, C, N_i)
                else:
                    assert mask_i.shape == (B, C, N_i), f"Mask shape {mask_i.shape} does not match expected [B={B}, C={C}, N_i={N_i}]"
                
                mask_valid = (mask_i == 1)
                mask_valid = mask_valid.view(B * C, N_i)
                attn_mask = torch.logical_or(mask_valid.unsqueeze(-1), mask_valid.unsqueeze(-2))
            
            x_normalized = pre_norm(x_in_reshaped)
            freqs = None
            if self.rope_intra:
                freqs = self.rope_intra.prepare_freqs_1d(
                    N_i, x_normalized.device, x_normalized.dtype
                )
                freqs = freqs.unsqueeze(0).unsqueeze(0)
            
            attn_out, attn_weights = attn_layer(
                x_normalized, x_normalized, x_normalized, attn_mask=attn_mask, freqs=freqs
            )
            x_out = x_in_reshaped + self.dropout(attn_out)
            x_out = post_norm(x_out)
            x_out = x_out.view(B, C, N_i, D)
            x_intra.append(x_out)
            attn_intra.append(attn_weights)
        
        attn_dict = {'intra': attn_intra}
        return x_intra, attn_dict

class MAGEChannelBlock(nn.Module):
    def __init__(
        self,
        num_granularities: int,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        output_attention: bool = False
    ):
        super().__init__()
        self.num_granularities = num_granularities
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.intra_attention = AttentionLayer(
            FullAttention(
                attention_dropout=dropout,
                output_attention=output_attention,
                use_rope=False  # Explicitly disable RoPE
            ),
            d_model=d_model,
            n_heads=n_heads
        )
        
        self.intra_pre_norm = nn.LayerNorm(d_model)
        self.intra_post_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_list: List[torch.Tensor],
        mask_list: Optional[List[torch.Tensor]] = None
    ):
        """
        Args:
            x_list: List of [B, C, N_i, D] tensors for each granularity
            mask_list: List of [B, C, N_i] or [C, N_i] where 1 = masked, 0 = non-masked for each scale
        Returns:
            x_out: List of [B, C, N_i, D] tensors
            attn_dict: Dictionary with 'intra' attention weights
        """
        if mask_list is None:
            mask_list = [None] * len(x_list)
        
        assert len(x_list) == self.num_granularities, \
            f"Expected {self.num_granularities} inputs, got {len(x_list)}"
        assert len(mask_list) == len(x_list), \
            f"Mask list length {len(mask_list)} does not match input list length {len(x_list)}"
        
        x_out = []
        attn_intra = []
        
        for x, mask in zip(x_list, mask_list):
            B, C, N_i, D = x.shape
            x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N_i, C, D)
            
            attn_mask = None
            if mask is not None:
                if mask.ndim == 2:
                    assert mask.shape == (C, N_i), f"Mask shape {mask.shape} does not match expected [C={C}, N_i={N_i}]"
                    mask = mask.unsqueeze(0).expand(B, C, N_i) 
                else:
                    assert mask.shape == (B, C, N_i), f"Mask shape {mask.shape} does not match expected [B={B}, C={C}, N_i={N_i}]"

                mask_per_time = mask.transpose(1, 2)
                mask_per_time = mask_per_time.reshape(B * N_i, C)
                query_mask = (mask_per_time == 1).unsqueeze(-1)
                kv_mask = (mask_per_time == 1).unsqueeze(-2)
                attn_mask = torch.logical_or(query_mask, kv_mask)
            
            x_normalized = self.intra_pre_norm(x_reshaped)
            
            attn_out, attn_weights = self.intra_attention(
                x_normalized, x_normalized, x_normalized,
                attn_mask=attn_mask,
                freqs=None  # No RoPE
            )

            x_out_i = x_reshaped + self.dropout(attn_out)
            x_out_i = self.intra_post_norm(x_out_i)

            x_out_i = x_out_i.view(B, N_i, C, D).permute(0, 2, 1, 3).contiguous()
            x_out.append(x_out_i)
            attn_intra.append(attn_weights)
        
        attn_dict = {'intra': attn_intra}
        return x_out, attn_dict

class ChannelTimeMAGEEncoderLayer(nn.Module):
    """Encoder layer with channel and time attention."""
    
    def __init__(
        self,
        num_granularities: int,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
        no_inter_attn: bool = False,
        use_channel_attn: bool = True,
        use_rope_time: bool = True
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.num_granularities = num_granularities
        self.d_model = d_model
        self.use_channel_attn = use_channel_attn
        
        # Time attention
        self.time_attention = MAGETimeBlock(
            num_granularities=num_granularities,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            output_attention=output_attention,
            no_inter_attn=no_inter_attn,
            use_rope_intra=use_rope_time,
            rope_dim_frac=1.0,
            rope_theta=10000
        )
        
        # Channel attention
        if self.use_channel_attn:
            self.channel_attention = MAGEChannelBlock(
                num_granularities=num_granularities,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                output_attention=output_attention
            )
        
        # Feed-forward network
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(
        self,
        x_list: List[torch.Tensor],
        mask_list: Optional[List[torch.Tensor]] = None
    ):
        """
        Args:
            x_list: List of [B, C, N_i, D] tensors for each granularity
            mask_list: List of [C, N_i] where 1 = masked, 0 = non-masked
        Returns:
            output: List of [B, C, N_i, D] tensors
            attns: Dict of attention weights {'time': {...}, 'channel': {...}}
        """
        if mask_list is None:
            mask_list = [None] * len(x_list)
        
        assert len(x_list) == self.num_granularities, \
            f"Expected {self.num_granularities} inputs, got {len(x_list)}"
        assert len(mask_list) == len(x_list), \
            f"Mask list length {len(mask_list)} does not match input list length {len(x_list)}"
        
        B, C = x_list[0].shape[:2]
        D = x_list[0].shape[-1]
        x_time_out, time_attns = self.time_attention(x_list, mask_list)
        
        # Residual connection
        x_list = [
            x + self.dropout(x_out)
            for x, x_out in zip(x_list, x_time_out)
        ]
        # Channel attention (if enabled)
        channel_attns = None
        if self.use_channel_attn:
            x_list, channel_attns = self.channel_attention(x_list, mask_list)
        
        # Normalize
        x_list = [self.norm1(x) for x in x_list]
        y_ffn = []
        for x in x_list:
            B_curr, C_curr, L_curr, D_curr = x.shape
            x_conv = x.view(B_curr * C_curr, L_curr, D_curr).transpose(1, 2)
            x_conv = self.dropout(self.activation(self.conv1(x_conv)))
            x_conv = self.dropout(self.conv2(x_conv))
            x_conv = x_conv.transpose(1, 2).view(B_curr, C_curr, L_curr, D_curr)
            y_ffn.append(x_conv)
        
        output = [self.norm2(x + y) for x, y in zip(x_list, y_ffn)]
        attns = {'time': time_attns, 'channel': channel_attns}
        return output, attns

class ChannelTimeMAGEEncoder(nn.Module):
    """Multi-scale Medformer encoder with stacked layers."""
    def __init__(
        self,
        num_granularities: int,
        depth: int,
        d_model: int,
        n_heads: int,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
        no_inter_attn: bool = False,
        use_channel_attn: bool = True,
        use_rope_time: bool = True
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.num_granularities = num_granularities
        self.depth = depth
        self.d_model = d_model

        self.layers = nn.ModuleList([
            ChannelTimeMAGEEncoderLayer(
                num_granularities=num_granularities,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                output_attention=output_attention,
                no_inter_attn=no_inter_attn,
                use_channel_attn=use_channel_attn,
                use_rope_time=use_rope_time
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(
        self,
        x_list: List[torch.Tensor],
        mask_list: Optional[List[torch.Tensor]] = None
    ):
        """
        Args:
            x_list: List of [B, C, N_i, D] tensors for each granularity
            mask_list: List of [C, N_i] where 1 = masked, 0 = non-masked
        Returns:
            x_list: List of [B, C, N_i, D] tensors
            all_attns: List of dicts [{'time': {...}, 'channel': {...}}, ...]
        """
        if mask_list is None:
            mask_list = [None] * len(x_list)

        assert len(x_list) == self.num_granularities, \
            f"Expected {self.num_granularities} inputs, got {len(x_list)}"
        assert len(mask_list) == len(x_list), \
            f"Mask list length {len(mask_list)} does not match input list length {len(x_list)}"

        all_attns = []
        for layer in self.layers:
            x_list, attns = layer(x_list, mask_list)
            all_attns.append(attns)

        x_list = [self.norm(x) for x in x_list]
        return x_list, all_attns

def get_module_params(module: Optional[nn.Module]) -> int:
    """Calculate the total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters()) if module is not None else 0

class PatchEmbedding(nn.Module):
    """
    Single-scale patch embedding with 1D convolution on p_i dimension and spectral branch.
    Input: [B, C, N_i, p_i] (pre-patchified)
    Output: [B, C, N_i, d_model]
    Uses 1D convolution to process p_i dimension, followed by global pooling and linear projection.
    """
    def __init__(
        self,
        patch_size: int, 
        num_patches: int, 
        d_model: int,
        k_temporal: int = 3,
        temporal_extra_convs: int = 2,
        dropout: float = 0.1,
        use_spectral: bool = True
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.d_model = d_model
        self.use_spectral = use_spectral

        if k_temporal > patch_size:
            raise ValueError(f"Kernel size {k_temporal} must not exceed patch_size {patch_size}")
        layers = []
        layers.append(
            nn.Conv1d(
                in_channels=1,
                out_channels=1, 
                kernel_size=k_temporal,
                stride=1,
                padding=k_temporal // 2,
            )
        )
        layers += [nn.GroupNorm(num_groups=1, num_channels=1), nn.GELU()]
        
        for _ in range(temporal_extra_convs):
            layers += [
                nn.Conv1d(
                    in_channels=1,
                    out_channels=1,
                    kernel_size=k_temporal,
                    stride=1,
                    padding=k_temporal // 2,
                ),
                nn.GroupNorm(num_groups=1, num_channels=1),
                nn.GELU(),
            ]
        
        self.temporal_conv = nn.Sequential(*layers)

        self.to_dmodel = nn.Sequential(
            nn.Linear(patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        if self.use_spectral:
            n_freq = patch_size // 2 + 1 
            self.spectral_proj = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.Dropout(dropout)
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, N_i, p_i] pre-patchified input
        Returns:
            [B, C, N_i, d_model]
        """
        if x.ndim != 4:
            raise ValueError(f"Expected [B, C, N_i, p_i], got {tuple(x.shape)}")
        
        B, C, N_i, p_i = x.shape
        if p_i != self.patch_size or N_i != self.num_patches:
            raise ValueError(f"Got p_i={p_i}, N_i={N_i} but expected patch_size={self.patch_size}, num_patches={self.num_patches}")

        x_temp = x.view(B * C * N_i, 1, p_i)
        x_temp = self.temporal_conv(x_temp)
        
        x_temp = self.to_dmodel(x_temp)

        x_temp = x_temp.view(B, C, N_i, self.d_model)

        if self.use_spectral:
            x_spec = x.view(B * C * N_i, p_i)
            spec = torch.fft.rfft(x_spec, dim=-1, norm='forward')
            spec = torch.abs(spec)
            x_spec = self.spectral_proj(spec)
            x_spec = x_spec.view(B, C, N_i, self.d_model)
            x_temp = x_temp + x_spec

        return x_temp

class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-scale patch embedding with scale and channel embeddings.
    Input: [B, C, T]
    Output: List of [B, C, N_i, d_model] for each scale
    """
    def __init__(
        self,
        time_length: int,
        num_channels: int,
        patch_sizes: List[int],
        d_model: int,
        k_temporal: int = 3,
        temporal_extra_convs: int = 2,
        dropout: float = 0.1,
        use_spectral: bool = True
    ):
        super().__init__()
        self.time_length = time_length
        self.num_channels = num_channels
        self.patch_sizes = patch_sizes
        self.d_model = d_model
        self.num_granularities = len(patch_sizes)
        self.use_spectral = use_spectral

        # Validate patch sizes
        for p in patch_sizes:
            if time_length % p != 0:
                raise ValueError(f"Patch size {p} must divide time_length {time_length}")
            if k_temporal > p:
                raise ValueError(f"Kernel size {k_temporal} must not exceed patch_size {p}")

        # Patch embedding modules for each scale
        self.patch_embedding_list = nn.ModuleList([
            PatchEmbedding(
                patch_size=p,
                num_patches=time_length // p,
                d_model=d_model,
                k_temporal=k_temporal,
                temporal_extra_convs=temporal_extra_convs,
                dropout=dropout,
                use_spectral=use_spectral
            ) for p in patch_sizes
        ])

        self.scale_embeds = nn.ParameterList([
            nn.Parameter(torch.empty(d_model)) for _ in range(self.num_granularities)
        ])

        self.channel_embeds = nn.Parameter(torch.empty(num_channels, d_model))

        for emb in self.scale_embeds:
            nn.init.trunc_normal_(emb, std=0.02)
        nn.init.trunc_normal_(self.channel_embeds, std=0.02)

    def forward(
        self,
        input: torch.Tensor,
        mask: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            input: [B, C, T]
            mask: List of [C, N_i] where 1 = masked, 0 = non-masked
        Returns:
            List of [B, C, N_i, d_model]
        """
        B, C, T = input.shape
        if C != self.num_channels or T != self.time_length:
            raise ValueError(f"Expected [B, {self.num_channels}, {self.time_length}], got {input.shape}")

        if mask is not None and len(mask) != self.num_granularities:
            raise ValueError(f"Expected {self.num_granularities} masks, got {len(mask)}")

        x_list = []
        for i, (p, embedder) in enumerate(zip(self.patch_sizes, self.patch_embedding_list)):
            N_i = T // p
            x = input.view(B, C, N_i, p)
            x_emb = embedder(x) 
            x_emb = x_emb + self.scale_embeds[i].view(1, 1, 1, self.d_model)
            x_emb = x_emb + self.channel_embeds.view(1, C, 1, self.d_model)
            x_list.append(x_emb)

        return x_list

def get_module_params(module: Optional[nn.Module]) -> int:
    """Calculate the total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters()) if module is not None else 0

class MAGEPredictor(nn.Module):
    """
    Predictor that operates on multi-scale encoded representations.
    Takes encoded visible tokens and predicts masked tokens at each scale.
    """
    def __init__(
        self,
        num_granularities: int,
        d_model: int,
        predictor_depth: int,
        n_heads: int,
        dropout: float = 0.1,
        output_attention: bool = False,
        no_inter_attn: bool = True,
        use_channel_attn: bool = True,
        use_rope_time: bool = False,
    ):
        super().__init__()
        self.num_granularities = num_granularities
        self.d_model = d_model
        
        # Learnable mask tokens for each scale
        self.mask_tokens = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, 1, d_model))
            for _ in range(num_granularities)
        ])
        
        # Predictor transformer
        self.predictor = ChannelTimeMAGEEncoder(
            num_granularities=num_granularities,
            depth=predictor_depth,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_model * 4,
            dropout=dropout,
            output_attention=output_attention,
            no_inter_attn=no_inter_attn,
            use_channel_attn=use_channel_attn,
            use_rope_time=use_rope_time,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for token in self.mask_tokens:
            nn.init.trunc_normal_(token, std=0.02)
    
    def forward(
        self,
        x_enc_list: List[torch.Tensor],
        mask_x_list: List[torch.Tensor],
        mask_y_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x_enc_list: List of [B, C, N_i, D] encoded visible tokens
            mask_x_list: List of [B, C, N_i] masks (1=visible, 0=masked)
            mask_y_list: List of [B, C, N_i] masks (1=masked, 0=visible)
        Returns:
            z_pred_list: List of [B, C, N_i, D] predictions for all tokens
            z_comb_list: List of [B, C, N_i, D] combined (visible + predicted)
        """
        z_comb_list = []
        
        for i, (x_enc, mask_x, mask_y) in enumerate(zip(x_enc_list, mask_x_list, mask_y_list)):
            B, C, N_i, D = x_enc.shape
            
            mask_token = self.mask_tokens[i].expand(B, C, N_i, D)
            
            mask_x_expanded = mask_x.unsqueeze(-1).expand(-1, -1, -1, D)
            z_comb = torch.where(mask_x_expanded.bool(), x_enc, mask_token)
            z_comb_list.append(z_comb)

        mask_for_attn = [1.0 - m for m in mask_x_list]
        z_pred_list, _ = self.predictor(z_comb_list, mask_list=mask_for_attn)
        
        return z_pred_list, z_comb_list

class MAGEAggregator(nn.Module):
    """
    Learns to weight different scales based on the input signal.
    Uses both temporal and frequency domain features.
    Takes raw input signal and produces scale weights.
    """
    def __init__(
        self,
        time_length: int,
        num_channels: int,
        patch_sizes: List[int],
        hidden_dim: int = 128,
        use_frequency: bool = True,
        freq_dim: int = 64,
        act: str = "gelu",
    ):
        super().__init__()
        self.time_length = time_length
        self.num_channels = num_channels
        self.patch_sizes = patch_sizes
        self.num_granularities = len(patch_sizes)
        self.use_frequency = use_frequency
        
        self.temporal_extractor = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        if self.use_frequency:
            self.n_freqs = time_length // 2 + 1
            
            self.freq_extractor = nn.Sequential(
                nn.Conv1d(num_channels, freq_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(freq_dim),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.Conv1d(freq_dim, freq_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(freq_dim),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            
            self.phase_extractor = nn.Sequential(
                nn.Conv1d(num_channels, freq_dim // 2, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(freq_dim // 2),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            
            fusion_dim = hidden_dim + freq_dim + freq_dim // 2
            self.cross_domain_fusion = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.Dropout(0.1),
            )
            
            mlp_input_dim = fusion_dim
        else:
            mlp_input_dim = hidden_dim
        
        self.weight_predictor = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim),
            nn.LayerNorm(mlp_input_dim),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_input_dim, mlp_input_dim // 2),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Linear(mlp_input_dim // 2, self.num_granularities),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [B, C, T] raw input signal
        Returns:
            weights: [B, num_granularities] scale weights (after softmax)
            freq_info: Dict with frequency domain information for visualization/analysis
        """
        B, C, T = x.shape
        
        temporal_features = self.temporal_extractor(x)
        temporal_features = temporal_features.squeeze(-1)
        
        freq_info = {}
        
        if self.use_frequency:
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')
            
            magnitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)
            
            freq_info['magnitude'] = magnitude.detach()
            freq_info['phase'] = phase.detach()
            
            mag_features = self.freq_extractor(magnitude)
            mag_features = mag_features.squeeze(-1)
            
            phase_features = self.phase_extractor(phase)
            phase_features = phase_features.squeeze(-1)
            
            combined_features = torch.cat([
                temporal_features,
                mag_features,
                phase_features
            ], dim=-1)
            
            fused_features = self.cross_domain_fusion(combined_features)
        else:
            fused_features = temporal_features
        
        logits = self.weight_predictor(fused_features)
        
        weights = F.softmax(logits, dim=-1)
        
        freq_info['scale_logits'] = logits.detach()
        return weights, freq_info
