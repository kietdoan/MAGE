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
    
    # Handle different frequency tensor shapes
    if freqs.ndim == 2:  # [L, rot_dim]
        freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, L, rot_dim]
    
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    
    assert rot_dim <= t.shape[-1], f"feature dim {t.shape[-1]} < rope dim {rot_dim}"
    assert rot_dim % 2 == 0, f"rot_dim must be even, got {rot_dim}"
    
    # Split tensor into parts: [left, middle (to rotate), right]
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]
    
    # Apply rotation: Re(z * e^(iθ)) = x*cos(θ) - y*sin(θ)
    #                 Im(z * e^(iθ)) = x*sin(θ) + y*cos(θ)
    # Combined: [x*cos - y*sin, x*sin + y*cos] = x*cos + rotate_half(x)*sin
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
        
        # Compute inverse frequencies: 1 / (theta^(2i/dim)) for i in [0, dim/2)
        # Shape: [dim//2]
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        
        # Register as parameter (learnable) or buffer (frozen)
        if learned_freq:
            self.freqs = nn.Parameter(inv_freq)
        else:
            self.register_buffer('freqs', inv_freq, persistent=True)
        
        # Cache for non-learned frequencies
        # Key: (sequence_length, device_str, dtype_str)
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
        # Position indices: [0, 1, 2, ..., L-1]
        pos = torch.arange(L, device=device, dtype=torch.float32) + offset
        
        # Apply position interpolation (for length extrapolation)
        pos = pos / self.interpolate_factor
        
        # Compute pos * freq for all position-frequency pairs
        # pos: [L], freqs: [dim//2] -> outer product: [L, dim//2]
        freqs_half = torch.outer(pos, self.freqs.to(device).to(torch.float32))
        
        # Duplicate frequencies for cos/sin pairs
        # [L, dim//2] -> [L, dim] where each freq appears twice
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
        # If frequencies are learned, always recompute (no caching)
        # This ensures gradients flow properly during training
        if self.learned_freq:
            return self._compute_freqs(L, device, dtype, offset)
        
        # For non-learned frequencies, use caching
        cache_key = ("freqs1d", L, str(device), str(dtype))
        
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            # Verify cached tensor is on correct device
            if cached.device == device:
                return cached
            else:
                # Move to correct device and update cache
                cached = cached.to(device)
                self.cache[cache_key] = cached
                return cached
        
        # Compute and cache
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
            # print("1111111111111111111111111111111111111111111111")
            queries = apply_rotary_emb(freqs, queries)
            keys = apply_rotary_emb(freqs, keys)
        
        # Compute attention scores
        scores = torch.einsum("bhle,bhse->bhls", queries, keys) * scale
        
        # Apply boolean mask (True = masked position)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1), float('-inf'))
        # print("scores: ", scores)
        # Detect fully masked rows (all -inf in scores)
        fully_masked = torch.all(scores == float('-inf'), dim=-1, keepdim=True)  # Shape: [B, H, L, 1]
        
        # Softmax and handle NaN
        A = torch.softmax(scores, dim=-1)
        # print("A1: ", A)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN with 0
        A = A.masked_fill(fully_masked, 0.0)  # Zero fully masked rows
        # print("A2: ", A)
        # Apply dropout
        A = self.dropout(A)
        
        # Compute output and zero for fully masked queries
        V = torch.einsum("bhls,bhse->bhle", A, values)
        V = V.masked_fill(fully_masked, 0.0)  # Zero output for fully masked queries
        
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

def test_attention_layer():
    # Parameters
    batch_size = 1
    seq_len_q = 4  # Query sequence length
    seq_len_kv = 4  # Key/Value sequence length
    d_model = 8
    n_heads = 4
    dropout = 0.0
    output_attention = True
    use_rope = True

    # Set random seed for reproducibility
    torch.manual_seed(41)

    # Create input tensors
    queries = torch.rand(batch_size, seq_len_q, d_model)
    keys = torch.rand(batch_size, seq_len_kv, d_model)
    values = torch.rand(batch_size, seq_len_kv, d_model)

    # Create separate masks for queries and keys/values
    query_mask = torch.randint(0, 2, (batch_size, seq_len_q)).bool()  # [B, seq_len_q], True = masked
    kv_mask = torch.randint(0, 2, (batch_size, seq_len_kv)).bool()  # [B, seq_len_kv], True = masked
    
    # Ensure at least one non-masked position per row
    for b in range(batch_size):
        if query_mask[b].all():
            idx = torch.randint(0, seq_len_q, (1,)).item()
            query_mask[b, idx] = False
        if kv_mask[b].all():
            idx = torch.randint(0, seq_len_kv, (1,)).item()
            kv_mask[b, idx] = False
    print("query mask: ", query_mask)
    print("kv mask: ", kv_mask)

    # Create attn_mask [B, seq_len_q, seq_len_kv]
    query_mask = query_mask.unsqueeze(-1)  # [B, seq_len_q, 1]
    kv_mask = kv_mask.unsqueeze(-2)  # [B, 1, seq_len_kv]
    attn_mask = torch.logical_or(query_mask, kv_mask)  # [B, seq_len_q, seq_len_kv]
    print("attn mask: ", attn_mask)
    # Create freqs for RoPE
    rope = RotaryEmbedding(dim=d_model // n_heads, theta=10000)
    freqs = rope.prepare_freqs_1d(seq_len_q, queries.device, queries.dtype)
    freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len_q, rot_dim]

    # Instantiate FullAttention and AttentionLayer
    attention = FullAttention(
        attention_dropout=dropout,
        output_attention=output_attention,
        use_rope=use_rope
    )
    attention_layer = AttentionLayer(
        attention=attention,
        d_model=d_model,
        n_heads=n_heads
    )

    # Forward pass
    output, attn_weights = attention_layer(
        queries=queries,
        keys=keys,
        values=values,
        attn_mask=attn_mask,
        freqs=freqs
    )

    # Print shapes and verify
    print("Input queries shape:", queries.shape)
    print("Input keys shape:", keys.shape)
    print("Input values shape:", values.shape)
    print("Query mask shape:", query_mask.shape)
    print("Key/Value mask shape:", kv_mask.shape)
    print("Attention mask shape:", attn_mask.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", attn_weights.shape if attn_weights is not None else None)
    
    # Verify attention weights sums
    print("Attention weights sum per row:", attn_weights.sum(dim=-1))
    print("Attention weights contain NaN:", torch.isnan(attn_weights).any())
    
    # Verify RoPE application
    if use_rope:
        print("RoPE enabled: queries and keys should be rotated")
    else:
        print("RoPE disabled")

# Run the test
# test_attention_layer()

class MedformerTimeBlock(nn.Module):
    """Medformer Time Block with intra-scale and inter-scale attention."""
    
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
        print("rope intra: ", self.rope_intra)
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
        
        # Intra-Scale Attention
        for i, (x_in, attn_layer, pre_norm, post_norm, mask_i) in enumerate(
            zip(x_list, self.intra_attentions, self.intra_pre_norms, self.intra_post_norms, mask_list)
        ):
            B, C, N_i, D = x_in.shape
            # Reshape x_in from [B, C, N_i, D] to [B*C, N_i, D]
            x_in_reshaped = x_in.view(B * C, N_i, D)
            # print("x: ", x_in.shape)
            # Around line 520-525 in the modelling code
            # if mask_i is not None:
            #     print("mask: ", mask_i.shape)
            # Prepare attention mask [B*C, N_i, N_i]
            attn_mask = None
            if mask_i is not None:
                # Handle both [C, N_i] and [B, C, N_i] mask shapes
                if mask_i.ndim == 2:
                    assert mask_i.shape == (C, N_i), f"Mask shape {mask_i.shape} does not match expected [C={C}, N_i={N_i}]"
                    mask_i = mask_i.unsqueeze(0).expand(B, C, N_i)  # [B, C, N_i]
                else:
                    assert mask_i.shape == (B, C, N_i), f"Mask shape {mask_i.shape} does not match expected [B={B}, C={C}, N_i={N_i}]"
                
                # mask_i: [B, C, N_i], 1 = masked, 0 = non-masked
                mask_valid = (mask_i == 1)  # [B, C, N_i], True = masked
                # Reshape to [B*C, N_i]
                mask_valid = mask_valid.view(B * C, N_i)
                # Create square boolean mask [B*C, N_i, N_i]
                attn_mask = torch.logical_or(mask_valid.unsqueeze(-1), mask_valid.unsqueeze(-2))  # [B*C, N_i, N_i]
            
            x_normalized = pre_norm(x_in_reshaped)
            freqs = None
            if self.rope_intra:
                freqs = self.rope_intra.prepare_freqs_1d(
                    N_i, x_normalized.device, x_normalized.dtype
                )
                freqs = freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, N_i, rot_dim]
            
            attn_out, attn_weights = attn_layer(
                x_normalized, x_normalized, x_normalized, attn_mask=attn_mask, freqs=freqs
            )
            x_out = x_in_reshaped + self.dropout(attn_out)
            x_out = post_norm(x_out)
            # Reshape back to [B, C, N_i, D]
            x_out = x_out.view(B, C, N_i, D)
            x_intra.append(x_out)
            attn_intra.append(attn_weights)
        
        # Inter-Scale Attention
        if self.inter_attention is not None:
            routers = []
            for i, (x, mask_i) in enumerate(zip(x_intra, mask_list)):
                B, C, N_i, D = x.shape
                x_reshaped = x.view(B * C, N_i, D)
                if mask_i is not None:
                    # Handle both [C, N_i] and [B, C, N_i] mask shapes
                    if mask_i.ndim == 2:
                        mask_i = mask_i.unsqueeze(0).expand(B, C, N_i)  # [B, C, N_i]
                    # mask_i: [B, C, N_i], 1 = masked, 0 = non-masked
                    mask_valid = (mask_i == 0).float()  # [B, C, N_i], 1 = non-masked, 0 = masked
                    mask_valid = mask_valid.view(B * C, N_i, 1)  # [B*C, N_i, 1]
                    router = (x_reshaped * mask_valid).sum(dim=1) / (mask_valid.sum(dim=1).clamp(min=1))  # [B*C, D]
                else:
                    router = x_reshaped.mean(dim=1)  # [B*C, D]
                routers.append(router)
            
            routers = torch.stack(routers, dim=1)  # [B*C, num_granularities, D]
            routers_normalized = self.inter_pre_norm(routers)
            routers_out, attn_inter = self.inter_attention(
                routers_normalized, routers_normalized, routers_normalized, attn_mask=None, freqs=None
            )
            routers_out = routers + self.dropout(routers_out)
            routers_out = self.inter_post_norm(routers_out)
            
            x_out = []
            for i, x in enumerate(x_intra):
                B, C, N_i, D = x.shape
                inter_info = routers_out[:, i, :].unsqueeze(1).expand(-1, N_i, -1)  # [B*C, N_i, D]
                inter_info = inter_info.view(B, C, N_i, D)  # [B, C, N_i, D]
                gate_input = torch.cat([x, inter_info], dim=-1)  # [B, C, N_i, 2*D]
                gate = self.inter_gates[i](gate_input)  # [B, C, N_i, D]
                x_fused = x + gate * self.dropout(inter_info)
                x_fused = self.inter_output_norms[i](x_fused)
                x_out.append(x_fused)
            
            attn_dict = {'intra': attn_intra, 'inter': attn_inter}
            return x_out, attn_dict
        else:
            attn_dict = {'intra': attn_intra}
            return x_intra, attn_dict

def test_medformer_time_block():
    # Parameters
    batch_size = 2
    num_granularities = 2
    channels = 3
    seq_lengths = [4, 6]  # Different N_i for each granularity
    d_model = 4
    n_heads = 1
    dropout = 0.1
    output_attention = True
    use_rope_intra = True

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create input x_list: [B, C, N_i, D]
    x_list = [
        torch.rand(batch_size, channels, seq_lengths[i], d_model)
        for i in range(num_granularities)
    ]

    # Create mask_list: [C, N_i], 1 = masked, 0 = non-masked
    mask_list = [
        torch.tensor([[1., 1., 0., 1.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.]]),
        torch.tensor([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.]])
    ]

    # Instantiate MedformerTimeBlock
    block = MedformerTimeBlock(
        num_granularities=num_granularities,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        output_attention=output_attention,
        no_inter_attn=False,
        use_rope_intra=use_rope_intra,
        rope_dim_frac=1.0,
        rope_theta=10000
    )

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nTotal number of parameters in MedformerTimeBlock: {total_params}")

    # Forward pass
    x_out, attn_dict = block(x_list, mask_list)

    # Print shapes and verify
    print("\nVerification:")
    print("Input x_list shapes:", [x.shape for x in x_list])
    print("Mask_list shapes:", [m.shape for m in mask_list])
    print("Output x_out shapes:", [x.shape for x in x_out])
    print("Intra attention shapes:", [a.shape if a is not None else None for a in attn_dict['intra']])
    if 'inter' in attn_dict:
        print("Inter attention shape:", attn_dict['inter'].shape if attn_dict['inter'] is not None else None)
    
    # Verify attention weights sums and RoPE application
    for i, attn in enumerate(attn_dict['intra']):
        if attn is not None:
            print(f"Intra attention {i} contains NaN:", torch.isnan(attn).any())
    if 'inter' in attn_dict and attn_dict['inter'] is not None:
        print("Inter attention contains NaN:", torch.isnan(attn_dict['inter']).any())
    
    # Verify RoPE
    if use_rope_intra:
        print("RoPE enabled for intra-attention")
    else:
        print("RoPE disabled for intra-attention")

# Run the test
# test_medformer_time_block()

class MedformerChannelBlock(nn.Module):
    """Medformer Channel Block for channel attention across multiple granularities."""
    
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
        
        # Intra-Scale Channel Attention
        for x, mask in zip(x_list, mask_list):
            B, C, N_i, D = x.shape
            # Reshape x from [B, C, N_i, D] to [B*N_i, C, D]
            x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * N_i, C, D)
            
            # Prepare attention mask [B*N_i, C, C]
            attn_mask = None
            if mask is not None:
                # Handle both [C, N_i] and [B, C, N_i] mask shapes
                if mask.ndim == 2:
                    assert mask.shape == (C, N_i), f"Mask shape {mask.shape} does not match expected [C={C}, N_i={N_i}]"
                    mask = mask.unsqueeze(0).expand(B, C, N_i)  # [B, C, N_i]
                else:
                    assert mask.shape == (B, C, N_i), f"Mask shape {mask.shape} does not match expected [B={B}, C={C}, N_i={N_i}]"
                
                # mask: [B, C, N_i], 1 = masked, 0 = non-masked
                mask_per_time = mask.transpose(1, 2)  # [B, N_i, C]
                mask_per_time = mask_per_time.reshape(B * N_i, C)  # [B*N_i, C]
                query_mask = (mask_per_time == 1).unsqueeze(-1)  # [B*N_i, C, 1]
                kv_mask = (mask_per_time == 1).unsqueeze(-2)  # [B*N_i, 1, C]
                attn_mask = torch.logical_or(query_mask, kv_mask)  # [B*N_i, C, C]
            
            # Normalize input
            x_normalized = self.intra_pre_norm(x_reshaped)
            
            # Apply channel attention
            attn_out, attn_weights = self.intra_attention(
                x_normalized, x_normalized, x_normalized,
                attn_mask=attn_mask,
                freqs=None  # No RoPE
            )
            
            # Residual connection and normalization
            x_out_i = x_reshaped + self.dropout(attn_out)
            x_out_i = self.intra_post_norm(x_out_i)
            
            # Reshape back to [B, C, N_i, D]
            x_out_i = x_out_i.view(B, N_i, C, D).permute(0, 2, 1, 3).contiguous()
            x_out.append(x_out_i)
            attn_intra.append(attn_weights)
        
        attn_dict = {'intra': attn_intra}
        return x_out, attn_dict

def test_medformer_channel_block():
    # Parameters
    batch_size = 2
    num_granularities = 2
    channels = 3
    seq_lengths = [4, 6]  # Different N_i for each granularity
    d_model = 4
    n_heads = 1
    dropout = 0.1
    output_attention = True

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create input x_list: List of [B, C, N_i, D]
    x_list = [
        torch.rand(batch_size, channels, seq_lengths[i], d_model)
        for i in range(num_granularities)
    ]

    # Create mask_list: List of [C, N_i], 1 = masked, 0 = non-masked
    mask_list = [
        torch.tensor([[1., 1., 0., 1.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.]]),
        torch.tensor([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.]])
    ]

    # Instantiate MedformerChannelBlock
    block = MedformerChannelBlock(
        num_granularities=num_granularities,
        d_model=d_model,
        n_heads=n_heads,
        dropout=dropout,
        output_attention=output_attention
    )

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\nTotal number of parameters in MedformerChannelBlock: {total_params}")

    # Forward pass
    x_out, attn_dict = block(x_list, mask_list)

    # Print shapes and verify
    print("\nVerification:")
    print("Input x_list shapes:", [x.shape for x in x_list])
    print("Mask_list shapes:", [m.shape for m in mask_list])
    print("Output x_out shapes:", [x.shape for x in x_out])
    print("Intra attention shapes:", [a.shape if a is not None else None for a in attn_dict['intra']])
    
    # Verify attention weights
    for i, attn in enumerate(attn_dict['intra']):
        if attn is not None:
            print(f"Intra attention {i} contains NaN:", torch.isnan(attn).any())

# Run the test
# test_medformer_channel_block()
import time
class ChannelTimeMedformerEncoderLayer(nn.Module):
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
        self.time_attention = MedformerTimeBlock(
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
            self.channel_attention = MedformerChannelBlock(
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
        # Time attention
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
        # Feed-forward network
        y_ffn = []
        for x in x_list:
            B_curr, C_curr, L_curr, D_curr = x.shape
            x_conv = x.view(B_curr * C_curr, L_curr, D_curr).transpose(1, 2)
            x_conv = self.dropout(self.activation(self.conv1(x_conv)))
            x_conv = self.dropout(self.conv2(x_conv))
            x_conv = x_conv.transpose(1, 2).view(B_curr, C_curr, L_curr, D_curr)
            y_ffn.append(x_conv)
        
        # Final residual + norm
        output = [self.norm2(x + y) for x, y in zip(x_list, y_ffn)]
        attns = {'time': time_attns, 'channel': channel_attns}
        return output, attns

def test_channel_time_medformer_encoder_layer():
    # Parameters
    batch_size = 1
    num_granularities = 2
    channels = 3
    seq_lengths = [4, 6]  # Different N_i for each granularity
    d_model = 4
    n_heads = 1
    d_ff = 16
    dropout = 0.1
    output_attention = True
    use_rope_time = True

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create input x_list: List of [B, C, N_i, D]
    x_list = [
        torch.rand(batch_size, channels, seq_lengths[i], d_model)
        for i in range(num_granularities)
    ]

    # Create mask_list: List of [C, N_i], 1 = masked, 0 = non-masked
    mask_list = [
        torch.tensor([[1., 1., 0., 1.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.]]),
        torch.tensor([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.]])
    ]

    # Instantiate ChannelTimeMedformerEncoderLayer
    encoder_layer = ChannelTimeMedformerEncoderLayer(
        num_granularities=num_granularities,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation="gelu",
        output_attention=output_attention,
        no_inter_attn=False,
        use_channel_attn=True,
        use_rope_time=use_rope_time
    )

    # Calculate total number of parameters
    total_params = sum(p.numel() for p in encoder_layer.parameters())
    print(f"\nTotal number of parameters in ChannelTimeMedformerEncoderLayer: {total_params}")

    # Forward pass
    output, attns = encoder_layer(x_list, mask_list)

    # Print shapes and verify
    print("\nVerification:")
    print("Input x_list shapes:", [x.shape for x in x_list])
    print("Mask_list shapes:", [m.shape for m in mask_list])
    print("Output shapes:", [x.shape for x in output])
    print("Time attention intra shapes:", [a.shape if a is not None else None for a in attns['time']['intra']])
    if 'inter' in attns['time']:
        print("Time attention inter shape:", attns['time']['inter'].shape if attns['time']['inter'] is not None else None)
    print("Channel attention intra shapes:", [a.shape if a is not None else None for a in attns['channel']['intra']])
    
    # Verify attention weights
    for i, attn in enumerate(attns['time']['intra']):
        if attn is not None:
            print(f"Time intra attention {i} contains NaN:", torch.isnan(attn).any())
    if 'inter' in attns['time'] and attns['time']['inter'] is not None:
        print("Time inter attention contains NaN:", torch.isnan(attns['time']['inter']).any())
    for i, attn in enumerate(attns['channel']['intra']):
        if attn is not None:
            print(f"Channel intra attention {i} contains NaN:", torch.isnan(attn).any())

class ChannelTimeMedformerEncoder(nn.Module):
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
            ChannelTimeMedformerEncoderLayer(
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
        print("ChannelTimeMedformerEncoder: ", use_rope_time)
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

def test_channel_time_medformer_encoder():
    # Parameters
    batch_size = 1
    num_granularities = 3
    channels = 3
    seq_lengths = [4, 6, 8]  # Different N_i for each granularity
    d_model = 64
    n_heads = 8
    d_ff = d_model*4
    depth = 3
    dropout = 0.1
    output_attention = True
    use_rope_time = True

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create input x_list: List of [B, C, N_i, D]
    x_list = [
        torch.rand(batch_size, channels, seq_lengths[i], d_model)
        for i in range(num_granularities)
    ]

    # Create mask_list: List of [C, N_i], 1 = masked, 0 = non-masked
    mask_list = [
        torch.tensor([[1., 1., 0., 1.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.]]),
        torch.tensor([[0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0.]]),
        torch.tensor([[0., 0., 0., 1., 0., 0., 1., 1.],
                      [0., 0., 0., 1., 0., 0., 1., 0.],
                      [0., 0., 0., 1., 0., 0., 0., 1.]])
    ]

    # Instantiate ChannelTimeMedformerEncoder
    encoder = ChannelTimeMedformerEncoder(
        num_granularities=num_granularities,
        depth=depth,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        activation="gelu",
        output_attention=output_attention,
        no_inter_attn=False,
        use_channel_attn=True,
        use_rope_time=use_rope_time
    )

    # Calculate total number of parameters
    total_params = get_module_params(encoder)
    print(f"\nTotal number of parameters in ChannelTimeMedformerEncoder: {total_params}")

    # Detailed parameter breakdown
    print("\nParameter Breakdown:")
    final_norm_params = get_module_params(encoder.norm)
    print(f"Final LayerNorm: {final_norm_params} parameters")
    for i, layer in enumerate(encoder.layers):
        print(f"\nLayer {i}:")
        # Time attention parameters
        time_attn_params = get_module_params(layer.time_attention)
        print(f"  MedformerTimeBlock: {time_attn_params} parameters")
        # Breakdown within MedformerTimeBlock
        intra_attn_params = sum(get_module_params(attn) for attn in layer.time_attention.intra_attentions)
        intra_norm_params = sum(
            get_module_params(norm)
            for norm_list in [layer.time_attention.intra_pre_norms, layer.time_attention.intra_post_norms]
            for norm in norm_list
        )
        rope_params = get_module_params(layer.time_attention.rope_intra)
        inter_attn_params = get_module_params(layer.time_attention.inter_attention)
        inter_norm_params = sum(
            get_module_params(norm)
            for norm in [layer.time_attention.inter_pre_norm, layer.time_attention.inter_post_norm]
            if norm is not None
        )
        inter_gate_params = sum(
            get_module_params(gate)
            for gate in layer.time_attention.inter_gates
            if layer.time_attention.inter_gates is not None
        )
        inter_output_norm_params = sum(
            get_module_params(norm)
            for norm in layer.time_attention.inter_output_norms
            if layer.time_attention.inter_output_norms is not None
        )
        print(f"    Intra Attention: {intra_attn_params} parameters")
        print(f"    Intra Norms (pre+post): {intra_norm_params} parameters")
        print(f"    RoPE: {rope_params} parameters")
        print(f"    Inter Attention: {inter_attn_params} parameters")
        print(f"    Inter Norms (pre+post): {inter_norm_params} parameters")
        print(f"    Inter Gates: {inter_gate_params} parameters")
        print(f"    Inter Output Norms: {inter_output_norm_params} parameters")

        # Channel attention parameters
        channel_attn_params = get_module_params(layer.channel_attention) if layer.use_channel_attn else 0
        print(f"  MedformerChannelBlock: {channel_attn_params} parameters")
        if layer.use_channel_attn:
            intra_attn_params = get_module_params(layer.channel_attention.intra_attention)
            intra_norm_params = sum(
                get_module_params(norm)
                for norm in [layer.channel_attention.intra_pre_norm, layer.channel_attention.intra_post_norm]
            )
            print(f"    Intra Attention: {intra_attn_params} parameters")
            print(f"    Intra Norms (pre+post): {intra_norm_params} parameters")

        # FFN parameters
        ffn_params = sum(get_module_params(conv) for conv in [layer.conv1, layer.conv2])
        norm_params = sum(get_module_params(norm) for norm in [layer.norm1, layer.norm2])
        print(f"  FFN (conv1+conv2): {ffn_params} parameters")
        print(f"  FFN Norms (norm1+norm2): {norm_params} parameters")
        layer_total = time_attn_params + channel_attn_params + ffn_params + norm_params
        print(f"  Total Layer {i} parameters: {layer_total} parameters")

    # Forward pass
    output, all_attns = encoder(x_list, mask_list)

    # Print shapes and verify
    print("\nVerification:")
    print("Input x_list shapes:", [x.shape for x in x_list])
    print("Mask_list shapes:", [m.shape for m in mask_list])
    print("Output shapes:", [x.shape for x in output])
    for i, attns in enumerate(all_attns):
        print(f"\nLayer {i} attention:")
        print("Time attention intra shapes:", [a.shape if a is not None else None for a in attns['time']['intra']])
        if 'inter' in attns['time']:
            print("Time attention inter shape:", attns['time']['inter'].shape if attns['time']['inter'] is not None else None)
        print("Channel attention intra shapes:", [a.shape if a is not None else None for a in attns['channel']['intra']])

        for j, attn in enumerate(attns['time']['intra']):
            if attn is not None:
                print(f"Layer {i} time intra attention {j} contains NaN:", torch.isnan(attn).any())
        if 'inter' in attns['time'] and attns['time']['inter'] is not None:
            print(f"Layer {i} time inter attention contains NaN:", torch.isnan(attn).any())
        for j, attn in enumerate(attns['channel']['intra']):
            if attn is not None:
                print(f"Layer {i} channel intra attention {j} contains NaN:", torch.isnan(attn).any())


class PatchEmbedding(nn.Module):
    """
    Single-scale patch embedding with 1D convolution on p_i dimension and spectral branch.
    Input: [B, C, N_i, p_i] (pre-patchified)
    Output: [B, C, N_i, d_model]
    Uses 1D convolution to process p_i dimension, followed by global pooling and linear projection.
    """
    def __init__(
        self,
        patch_size: int,  # p_i
        num_patches: int,  # N_i
        d_model: int,
        k_temporal: int = 3,  # Kernel size cho 1D conv
        temporal_extra_convs: int = 2,  # Số layer conv bổ sung
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

        # Temporal branch: 1D Convolution stack trên p_i dimension
        layers = []
        # Layer đầu tiên: mỗi channel xử lý độc lập
        layers.append(
            nn.Conv1d(
                in_channels=1,  # Mỗi channel là 1D signal
                out_channels=1,  # Giữ output channels = 1 để đơn giản
                kernel_size=k_temporal,
                stride=1,
                padding=k_temporal // 2,  # Padding để giữ kích thước
            )
        )
        layers += [nn.GroupNorm(num_groups=1, num_channels=1), nn.GELU()]
        
        # Các layer conv bổ sung (nếu có)
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

        # Project trực tiếp sang d_model
        self.to_dmodel = nn.Sequential(
            nn.Linear(patch_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Spectral branch: Linear projection của FFT magnitudes
        if self.use_spectral:
            n_freq = patch_size // 2 + 1  # Output size của rfft
            self.spectral_proj = nn.Sequential(
                nn.Linear(n_freq, d_model),
                nn.Dropout(dropout)
            )

        # Initialize weights
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

        # Temporal branch: 1D Convolution trên p_i dimension
        x_temp = x.view(B * C * N_i, 1, p_i)
        x_temp = self.temporal_conv(x_temp)  # [B*C*N_i, 1, p_i]
        
        # Project trực tiếp sang d_model
        x_temp = self.to_dmodel(x_temp)  # [B*C*N_i, d_model]
        
        # Reshape back: [B, C, N_i, d_model]
        x_temp = x_temp.view(B, C, N_i, self.d_model)

        # Spectral branch
        if self.use_spectral:
            x_spec = x.view(B * C * N_i, p_i)  # [B*C*N_i, p_i]
            spec = torch.fft.rfft(x_spec, dim=-1, norm='forward')  # [B*C*N_i, n_freq]
            spec = torch.abs(spec)  # Magnitude
            x_spec = self.spectral_proj(spec)  # [B*C*N_i, d_model]
            x_spec = x_spec.view(B, C, N_i, self.d_model)  # [B, C, N_i, d_model]
            x_temp = x_temp + x_spec  # Combine branches

        return x_temp  # [B, C, N_i, d_model]

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

        # Scale embeddings: one for each granularity
        self.scale_embeds = nn.ParameterList([
            nn.Parameter(torch.empty(d_model)) for _ in range(self.num_granularities)
        ])

        # Channel embeddings: one for each channel
        self.channel_embeds = nn.Parameter(torch.empty(num_channels, d_model))

        # Initialize embeddings
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
            # Reshape input to [B, C, N_i, p_i]
            x = input.view(B, C, N_i, p)
            x_emb = embedder(x)  # [B, C, N_i, d_model]
            # Add scale embedding
            x_emb = x_emb + self.scale_embeds[i].view(1, 1, 1, self.d_model)
            # Add channel embedding
            x_emb = x_emb + self.channel_embeds.view(1, C, 1, self.d_model)
            x_list.append(x_emb)

        return x_list

def get_module_params(module: Optional[nn.Module]) -> int:
    """Calculate the total number of parameters in a module."""
    return sum(p.numel() for p in module.parameters()) if module is not None else 0

