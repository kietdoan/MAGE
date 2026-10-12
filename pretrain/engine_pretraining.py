from __future__ import annotations

import copy
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple, List

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from utils import CosineWDSchedule, grad_logger
from modelling_pretraining import (
    ChannelTimeMedformerEncoder,
    MultiScalePatchEmbedding,
)
class CollapseMonitor:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    @torch.no_grad()
    def check(self, z: torch.Tensor):
        """
        Expects z in shape [Batch, Time, Channel, Dim]
        """
        if z.ndim != 4: return {}
        
        # Calculate standard deviations
        std_batch = z.std(dim=0).mean()    # Batch diversity
        std_time = z.std(dim=1).mean()     # Temporal diversity
        std_channel = z.std(dim=2).mean()  # Channel diversity
        
        # Effective Rank Calculation (SVD)
        # Flatten to [N_tokens, Dim]
        z_flat = z.reshape(-1, z.shape[-1])
        z_centered = z_flat - z_flat.mean(dim=0)
        
        # Subsample if too large to save speed
        if z_flat.shape[0] > 2000:
            idx = torch.randperm(z_flat.shape[0], device=z.device)[:2000]
            z_centered = z_centered[idx]
            
        try:
            _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
            S_norm = S / (S.sum() + 1e-6)
            entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-6))
            eff_rank = torch.exp(entropy)
        except RuntimeError:
            eff_rank = torch.tensor(0.0)

        return {
            "collapse/std_batch": std_batch.item(),
            "collapse/std_time": std_time.item(),
            "collapse/std_channel": std_channel.item(),
            "collapse/eff_rank": eff_rank.item(),
            "debug/target_std": z.std().item()
        }

# ---------------------------------------------------------------------
# Canonical channel ordering (58 channels; matches img_size=(58, ...))
# ---------------------------------------------------------------------
DEFAULT_CHANNELS: List[str] = [
    "FP1","FPZ","FP2","AF3","AF4",
    "F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO3","POZ","PO4","PO8","O1","OZ","O2",
]

DEFAULT_CH_19: List[str] = [
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4",
    "T8","P7","P3","PZ","P4","P8","O1","O2"
]

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def _maybe_ln(x: torch.Tensor, use_ln: bool) -> torch.Tensor:
    """Apply layer normalization if enabled."""
    return F.layer_norm(x, (x.size(-1),)) if use_ln else x

def _gather_tokens_multiscale(
    x_list: List[torch.Tensor], 
    mask_list: List[torch.Tensor]
) -> torch.Tensor:
    """
    Gather selected tokens from multiple scales.
    Returns: [Total_selected_tokens_across_batch_and_scales, D]
    """
    gathered = []
    for x, mask in zip(x_list, mask_list):
        # x: [B, C, N, D], mask: [B, C, N]
        mask_exp = mask.unsqueeze(-1).expand_as(x)   # [B, C, N, D]
        selected = x[mask_exp.bool()]                # [num_selected, D]
        if selected.numel() > 0:
            gathered.append(selected)
    
    if len(gathered) == 0:
        return x_list[0].new_zeros((0, x_list[0].size(-1)))
    
    return torch.cat(gathered, dim=0)  # [Total, D]

# ---------------------------------------------------------------------
# Multi-Scale Predictor
# ---------------------------------------------------------------------
class MultiScalePredictor(nn.Module):
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
        self.predictor = ChannelTimeMedformerEncoder(
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
        
        # For each scale, combine visible tokens with mask tokens
        for i, (x_enc, mask_x, mask_y) in enumerate(zip(x_enc_list, mask_x_list, mask_y_list)):
            B, C, N_i, D = x_enc.shape
            
            # Create mask token tensor
            mask_token = self.mask_tokens[i].expand(B, C, N_i, D)
            
            # Combine: use encoded tokens where visible, mask tokens where masked
            mask_x_expanded = mask_x.unsqueeze(-1).expand(-1, -1, -1, D)
            z_comb = torch.where(mask_x_expanded.bool(), x_enc, mask_token)
            z_comb_list.append(z_comb)
        
        mask_for_attn = [1.0 - m for m in mask_x_list]  # Invert: 1=masked, 0=visible
        z_pred_list, _ = self.predictor(z_comb_list, mask_list=None)
        
        return z_pred_list, z_comb_list


# ---------------------------------------------------------------------
# Multi-Scale Aggregator (learns scale weights from input signal)
# ---------------------------------------------------------------------
class MultiScaleAggregator(nn.Module):
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
        
        # Temporal feature extractor
        self.temporal_extractor = nn.Sequential(
            nn.Conv1d(num_channels, hidden_dim, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU() if act == "gelu" else nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global pooling -> [B, hidden_dim, 1]
        )
        
        # Frequency domain feature extractor
        if self.use_frequency:
            # Compute number of frequency bins (rfft output)
            self.n_freqs = time_length // 2 + 1
            
            # Frequency feature extractor: processes magnitude and phase
            self.freq_extractor = nn.Sequential(
                # Input: [B, C, n_freqs] for magnitude
                nn.Conv1d(num_channels, freq_dim, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(freq_dim),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.Conv1d(freq_dim, freq_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(freq_dim),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Global pooling -> [B, freq_dim, 1]
            )
            
            # Phase feature extractor (separate branch for phase information)
            self.phase_extractor = nn.Sequential(
                nn.Conv1d(num_channels, freq_dim // 2, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm1d(freq_dim // 2),
                nn.GELU() if act == "gelu" else nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # Global pooling -> [B, freq_dim//2, 1]
            )
            
            # Cross-domain attention to fuse temporal and frequency features
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
        
        # MLP to produce scale weights from fused features
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
        
        # Extract temporal features
        temporal_features = self.temporal_extractor(x)  # [B, hidden_dim, 1]
        temporal_features = temporal_features.squeeze(-1)  # [B, hidden_dim]
        
        freq_info = {}
        
        if self.use_frequency:
            # Compute FFT (using rfft for real signals)
            # rfft output: [B, C, n_freqs] complex
            x_fft = torch.fft.rfft(x, dim=-1, norm='ortho')  # Orthogonal normalization
            
            # Extract magnitude and phase
            magnitude = torch.abs(x_fft)  # [B, C, n_freqs]
            phase = torch.angle(x_fft)    # [B, C, n_freqs]
            
            # Store for potential visualization
            freq_info['magnitude'] = magnitude.detach()
            freq_info['phase'] = phase.detach()
            
            # Extract magnitude features
            mag_features = self.freq_extractor(magnitude)  # [B, freq_dim, 1]
            mag_features = mag_features.squeeze(-1)  # [B, freq_dim]
            
            # Extract phase features
            phase_features = self.phase_extractor(phase)  # [B, freq_dim//2, 1]
            phase_features = phase_features.squeeze(-1)  # [B, freq_dim//2]
            
            # Concatenate all features
            combined_features = torch.cat([
                temporal_features,
                mag_features,
                phase_features
            ], dim=-1)  # [B, hidden_dim + freq_dim + freq_dim//2]
            
            # Cross-domain fusion
            fused_features = self.cross_domain_fusion(combined_features)
        else:
            fused_features = temporal_features
        
        # Predict scale weights
        logits = self.weight_predictor(fused_features)  # [B, num_granularities]
        
        # Softmax to get normalized weights
        weights = F.softmax(logits, dim=-1)  # [B, num_granularities]
        
        # Store logits for analysis
        freq_info['scale_logits'] = logits.detach()
        return weights, freq_info


# ---------------------------------------------------------------------
# Multi-Scale Decoder (decodes with learned weights from aggregator)
# ---------------------------------------------------------------------
class MultiScaleDecoder(nn.Module):
    """
    Decodes multi-scale representations to single signal using provided weights.
    """
    def __init__(
        self,
        time_length: int,
        num_channels: int,
        patch_sizes: List[int],
        d_model: int,
        hidden_ratio: float = 2.0,
        act: str = "gelu",
    ):
        super().__init__()
        self.time_length = time_length
        self.num_channels = num_channels
        self.patch_sizes = patch_sizes
        self.num_granularities = len(patch_sizes)
        self.d_model = d_model
        
        # Projection heads for each scale
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU() if act == "gelu" else nn.ReLU(),
            ) for _ in patch_sizes
        ])
        
        # Decoder heads for each scale (output to respective patch_size)
        self.decoder_heads = nn.ModuleList()
        hidden_dim = int(d_model * hidden_ratio) if hidden_ratio > 0 else None
        for p in patch_sizes:
            if hidden_ratio > 0:
                self.decoder_heads.append(
                    nn.Sequential(
                        nn.Linear(d_model, hidden_dim),
                        nn.LayerNorm(hidden_dim),
                        nn.GELU() if act == "gelu" else nn.ReLU(),
                        nn.Linear(hidden_dim, p),
                    )
                )
            else:
                self.decoder_heads.append(nn.Linear(d_model, p))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x_list: List[torch.Tensor],
        scale_weights: torch.Tensor,
        mask_list: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_list: List of [B, C, N_i, D] tensors for each scale
            scale_weights: [B, num_granularities] weights for each scale
            mask_list: Optional list of [B, C, N_i] masks (1=reconstruct, 0=ignore)
        Returns:
            [B, C, T] reconstructed signal
        """
        B, C = x_list[0].shape[:2]
        
        # Process each scale
        processed_scales = []
        for i, (x, proj, decoder_head) in enumerate(
            zip(x_list, self.scale_projections, self.decoder_heads)
        ):
            # Project features
            x_proj = proj(x)  # [B, C, N_i, D]
            
            # Decode to patch representation
            x_patches = decoder_head(x_proj)  # [B, C, N_i, patch_size_i]
            
            # Apply mask if provided (1=reconstruct, 0=ignore)
            if mask_list is not None:
                mask = mask_list[i].unsqueeze(-1)  # [B, C, N_i, 1]
                x_patches = x_patches * mask
            
            # Reshape to full time resolution
            x_recon = x_patches.reshape(B, C, self.time_length)
            
            # Weight by learned scale importance (per-sample weights)
            # scale_weights: [B, num_granularities] -> [B, 1, 1]
            w = scale_weights[:, i].view(B, 1, 1)
            x_recon = x_recon * w
            processed_scales.append(x_recon)
        
        # Aggregate scales
        x_fused = torch.stack(processed_scales, dim=0).sum(dim=0)  # [B, C, T]
        
        return x_fused


# ---------------------------------------------------------------------
# Multi-Scale Lightning Module
# ---------------------------------------------------------------------
class LitMAGE(pl.LightningModule):
    """
    Multi-scale pretraining engine with JEPA-style learning.
    - Encoder: Multi-scale transformer operating on different patch sizes
    - Predictor: Predicts masked tokens at each scale
    - Decoder: Aggregates multi-scale representations into single signal
    """
    
    def __init__(
        self,
        *,
        # Data config
        time_length: int = 6000,
        num_channels: int = 19,
        patch_sizes: List[int] = [200, 300, 600],
        channels: Optional[List[str]] = None,
        montage: Optional[str] = None,
        mask_ratio: float = 0.5,
        # Model config
        d_model: int = 256,
        encoder_depth: int = 6,
        predictor_depth: int = 3,
        encoder_heads: int = 8,
        predictor_heads: int = 4,
        k_temporal: int = 3,
        temporal_extra_convs: int = 2,
        use_spectral: bool = True,
        dropout: float = 0.1,
        output_attention: bool = False,
        no_inter_attn_enc: bool = False,
        no_inter_attn_pred: bool = True,
        use_channel_attn: bool = True,
        use_rope_time: bool = True,
        # Decoder config
        decoder_hidden_ratio: float = 2.0,
        # Training schedules
        steps_per_epoch: int = 1000,
        max_epochs: int = 50,
        pretrain_max_lr: float = 3e-4,
        ref_wd: float = 5e-2,
        final_wd: float = 5e-2,
        ema_start_end: Tuple[float, float] = (0.996, 0.9995),
        # Loss config
        USE_LOSS_R: bool = True,
        USE_LOSS_A: bool = True,
        USE_ALIGN_VIS: bool = True,
        USE_ALIGN_MASK: bool = True,
        USE_LN: bool = True,
        lambda_recon: float = 10.0,
        lambda_align_vis: float = 1.0,
        lambda_align_mask: float = 1.0,
        tag_name: str = "base",
        loss_schedule: str = "fixed",
        # [NEW] Aggregator Regularization
        agg_noise_std: float = 0.0,   # Std of noise added to aggregator weights
        scale_dropout: float = 0.0,   # Probability to drop a scale during training
        entropy_lambda: float = 0.0,  # Regularization to penalize low entropy
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Validate inputs
        for p in patch_sizes:
            if time_length % p != 0:
                raise ValueError(f"time_length {time_length} must be divisible by patch_size {p}")
        
        # Store config
        self.time_length = time_length
        self.num_channels = num_channels
        self.patch_sizes = patch_sizes
        self.num_granularities = len(patch_sizes)
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        # Flags
        self.USE_LOSS_A = USE_LOSS_A
        self.USE_LOSS_R = USE_LOSS_R
        self.USE_LN = USE_LN
        self.USE_ALIGN_MASK = USE_ALIGN_MASK
        self.USE_ALIGN_VIS = USE_ALIGN_VIS
        
        # Loss weights
        self.lambda_recon = lambda_recon
        self.lambda_align_vis = lambda_align_vis
        self.lambda_align_mask = lambda_align_mask
        self.tag_name = tag_name
        self.loss_schedule = loss_schedule
        
        # [NEW] Aggregator Regularization Config
        self.agg_noise_std = agg_noise_std
        self.scale_dropout = scale_dropout
        self.entropy_lambda = entropy_lambda

        # Training config
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.pretrain_max_lr = pretrain_max_lr
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.ema_start, self.ema_end = ema_start_end
        
        # Channel setup
        if channels is not None:
            chans = channels
        else:
            chans = DEFAULT_CH_19 if (montage or "").strip() == "19" else DEFAULT_CHANNELS
        
        # ----------------- Models ----------------- #
        
        # Patch embedding
        self.patch_embed = MultiScalePatchEmbedding(
            time_length=time_length,
            num_channels=num_channels,
            patch_sizes=patch_sizes,
            d_model=d_model,
            k_temporal=k_temporal,
            temporal_extra_convs=temporal_extra_convs,
            dropout=dropout,
            use_spectral=use_spectral,
        )
        
        # Student encoder
        self.encoder = ChannelTimeMedformerEncoder(
            num_granularities=self.num_granularities,
            depth=encoder_depth,
            d_model=d_model,
            n_heads=encoder_heads,
            d_ff=d_model * 4,
            dropout=dropout,
            output_attention=output_attention,
            no_inter_attn=no_inter_attn_enc,
            use_channel_attn=use_channel_attn,
            use_rope_time=use_rope_time,
        )
        
        # EMA teacher (frozen copy of encoder)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        
        # Predictor
        self.predictor = MultiScalePredictor(
            num_granularities=self.num_granularities,
            d_model=d_model,
            predictor_depth=predictor_depth,
            n_heads=predictor_heads,
            dropout=dropout,
            output_attention=output_attention,
            no_inter_attn=no_inter_attn_pred,
            use_channel_attn=use_channel_attn,
            use_rope_time=False,  # Typically disabled in predictor
        )
        print("d model: ", d_model)
        # Aggregation decoder
        self.aggregator = MultiScaleAggregator(
            time_length=time_length,
            num_channels=num_channels,
            patch_sizes=patch_sizes,
            hidden_dim=d_model//2,
            use_frequency=True,  # Enable frequency domain features
            freq_dim=d_model//4,
            act="gelu",
        )
        
        self.decoder = MultiScaleDecoder(
            time_length=time_length,
            num_channels=num_channels,
            patch_sizes=patch_sizes,
            d_model=d_model,
            hidden_ratio=decoder_hidden_ratio,
            act="gelu",
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # --- Collapse Monitor ---
        self.monitor = CollapseMonitor(feature_dim=d_model)

        # Schedulers (set in configure_optimizers)
        self.wd_scheduler = None
        self.momentum_scheduler = None
        
        # Schedule bases for loss weighting
        self._sched_bases: Dict[str, Dict[str, float]] = {
            "nano":   dict(lambda_recon=5.0, vis_start=1.00, vis_end=1.00, hold_frac=1.00, warm_frac=0.00),
            "tiny":   dict(lambda_recon=5.0, vis_start=1.00, vis_end=1.00, hold_frac=1.00, warm_frac=0.00),
            "little": dict(lambda_recon=5.0, vis_start=0.70, vis_end=0.30, hold_frac=0.30, warm_frac=0.00),
            "base":   dict(lambda_recon=5.0, vis_start=0.15, vis_end=0.00, hold_frac=0.10, warm_frac=0.00),
            "large":  dict(lambda_recon=5.0, vis_start=0.10, vis_end=0.00, hold_frac=0.05, warm_frac=0.05),
        }
    
    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def _sample_masks_multiscale(
        self,
        mask_ratio: float,
        device: torch.device,
        batch_size: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate hierarchical masks for each scale.
        If a token is masked at coarse scale, it's also masked at finer scales.
        
        Returns (mask_x_list, mask_y_list) where:
            mask_x_list: List of [B, C, N_i] visible masks (1=visible, 0=masked)
            mask_y_list: List of [B, C, N_i] masked masks (1=masked, 0=visible)
        """
        C = self.num_channels
        
        # Start with the coarsest scale (largest patch size)
        coarsest_patch_size = max(self.patch_sizes)
        coarsest_num_patches = self.time_length // coarsest_patch_size
        
        # Generate mask at coarsest scale using Bernoulli sampling
        # mask: 1 = masked, 0 = visible
        coarse_mask = torch.zeros(
            (batch_size, C, coarsest_num_patches), 
            dtype=torch.float32, 
            device=device
        )
        coarse_mask = coarse_mask.bernoulli_(mask_ratio)  # [B, C, N_coarse]
        
        # Build mask list from coarse to fine
        mask_y_list_temp = [coarse_mask]
        
        # Find the index of the coarsest scale
        coarsest_idx = self.patch_sizes.index(coarsest_patch_size)
        
        # Propagate mask to finer scales
        for i in range(len(self.patch_sizes) - 1, -1, -1):
            if i == coarsest_idx:
                continue  # Skip the coarsest scale (already added)
            
            patch_size = self.patch_sizes[i]
            num_patches = self.time_length // patch_size
            
            # Calculate mapping from fine scale to coarse scale
            # Each position in fine scale maps to a position in coarse scale
            indices = torch.arange(num_patches, device=device) * patch_size // coarsest_patch_size
            indices = indices.clamp(0, coarsest_num_patches - 1)  # Ensure valid indices
            
            # Propagate mask from coarse to fine
            fine_mask = coarse_mask[:, :, indices]  # [B, C, N_fine]
            mask_y_list_temp.append(fine_mask)
        
        # Reverse to get mask list from fine to coarse (matching patch_sizes order)
        mask_y_list_temp.reverse()
        
        # Reorder to match self.patch_sizes order
        # Create mapping from patch_size to mask
        patch_to_mask = {}
        # print("self.patch_size: ", self.patch_sizes)
        sorted_patches = sorted(self.patch_sizes, reverse=True)  # Coarse to fine
        # print("self.patch_size: ", sorted_patches)
        for patch_size, mask in zip(sorted_patches, mask_y_list_temp[:len(sorted_patches)]):
            patch_to_mask[patch_size] = mask
        
        # Build final mask lists in the correct order
        mask_y_list = []
        mask_x_list = []
        for patch_size in self.patch_sizes:
            mask_y = patch_to_mask[patch_size]  # 1 = masked, 0 = visible
            mask_x = 1.0 - mask_y  # 1 = visible, 0 = masked
            mask_y_list.append(mask_y)
            mask_x_list.append(mask_x)
            # print("mask x: ", mask_x.shape)
            # print("mask y: ", mask_y.shape)
        mask_y_list.reverse()
        mask_x_list.reverse()
        return mask_x_list, mask_y_list
    
    # ------------------------------------------------------------------
    # Teacher path
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward_target(
        self,
        x: torch.Tensor,
        mask_y_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Teacher forward pass (no masking in attention).
        Args:
            x: [B, C, T] input signal
            mask_y_list: List of [B, C, N_i] masks for target extraction
        Returns:
            h_list: List of [B, C, N_i, D] teacher representations
            y_tgt: [B, C, T] target signal (same as input)
        """
        # Patch embed
        # mean_per_channel = x.mean(dim=(0, 2))
        # std_per_channel = x.std(dim=(0, 2))
        # print("Input mean per channel:", mean_per_channel)
        # print("Input std per channel:", std_per_channel)
        # print("shape: ", x.shape)
        x_list = self.patch_embed(x)  # List of [B, C, N_i, D]
        
        # Teacher encoder (no masking)
        h_list, _ = self.target_encoder(x_list, mask_list=None)
        
        return h_list, x
    
    # ------------------------------------------------------------------
    # Student path
    # ------------------------------------------------------------------
    def forward_context(
        self,
        x: torch.Tensor,
        mask_x_list: List[torch.Tensor],
        mask_y_list: List[torch.Tensor],
        scale_weights: Optional[torch.Tensor] = None, # [NEW] Optional Argument
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        """
        Student forward pass with masked encoding and prediction.
        Args:
            x: [B, C, T] input signal
            mask_x_list: List of [B, C, N_i] visible masks (1=visible, 0=masked)
            mask_y_list: List of [B, C, N_i] masked masks (1=masked, 0=visible)
            scale_weights: [Optional] Pre-calculated scale weights (e.g. with noise)
        Returns:
            z_enc_list: List of [B, C, N_i, D] encoded representations
            z_pred_list: List of [B, C, N_i, D] predicted representations
            z_comb_list: List of [B, C, N_i, D] combined representations
            r: [B, C, T] reconstructed signal
        """
        # Patch embed
        # [MODIFIED] Use provided weights if available, else compute
        if scale_weights is None:
            scale_weights, freq_info = self.aggregator(x)
        
        # print("scale weights: ", scale_weights)
        x_list = self.patch_embed(x)  # List of [B, C, N_i, D]
        
        # Student encoder with masking (convert to attention mask: 1=masked, 0=visible)
        mask_for_attn = [1.0 - m for m in mask_x_list]
        z_enc_list, _ = self.encoder(x_list, mask_list=mask_for_attn)
        
        # Predictor
        z_pred_list, z_comb_list = self.predictor(z_enc_list, mask_x_list, mask_y_list)
        
        # Aggregate and decode
        r = self.decoder(z_pred_list, scale_weights)  # [B, C, T]
        
        return z_enc_list, z_pred_list, z_comb_list, r
    
    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_1_to_0(x: float) -> float:
        x = max(0.0, min(1.0, x))
        return 0.5 * (1.0 + math.cos(math.pi * x))
    
    def _effective_loss_weights(self) -> Dict[str, float]:
        """Compute effective loss weights based on schedule."""
        if self.loss_schedule == "fixed":
            return {
                "lambda_recon": self.lambda_recon if self.USE_LOSS_R else 0.0,
                "lambda_align_vis": (self.lambda_align_vis if (self.USE_LOSS_A and self.USE_ALIGN_VIS) else 0.0),
                "lambda_align_mask": (self.lambda_align_mask if (self.USE_LOSS_A and self.USE_ALIGN_MASK) else 0.0),
            }
        
        # Scale-aware cosine schedule
        base = self._sched_bases.get(self.tag_name, self._sched_bases["base"])
        T = max(1, int(self.max_epochs * self.steps_per_epoch))
        s = max(0, min(self.global_step, T))
        
        hold_steps = int(T * base.get("hold_frac", 0.0))
        warm_steps = int(T * base.get("warm_frac", 0.0))
        decay_start = max(hold_steps, warm_steps)
        decay_span = max(1, T - decay_start)
        
        # Visible weight decay
        if s <= decay_start:
            vis_w = base["vis_start"]
        else:
            decay_prog = (s - decay_start) / decay_span
            cos = self._cosine_1_to_0(decay_prog)
            vis_w = base["vis_end"] + (base["vis_start"] - base["vis_end"]) * cos
        
        # Masked weight (constant)
        mask_w = float(self.lambda_align_mask)
        
        # Reconstruction weight (constant)
        recon_w = float(base["lambda_recon"])
        
        # Apply toggles
        vis_w = vis_w if (self.USE_LOSS_A and self.USE_ALIGN_VIS) else 0.0
        mask_w = mask_w if (self.USE_LOSS_A and self.USE_ALIGN_MASK) else 0.0
        recon_w = recon_w if self.USE_LOSS_R else 0.0
        
        return {
            "lambda_recon": float(recon_w),
            "lambda_align_vis": float(vis_w),
            "lambda_align_mask": float(mask_w),
        }
    
    def _compute_losses(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Compute all losses."""
        device = x.device
        B = x.size(0)

        # Sample hierarchical masks
        mask_x_list, mask_y_list = self._sample_masks_multiscale(
            mask_ratio=self.mask_ratio,
            device=device,
            batch_size=B,
        )

        # -------------------------------------------------------
        # [MODIFIED] Aggregator Regularization logic
        # -------------------------------------------------------
        # Get adaptive scale weights from aggregator
        scale_weights, freq_info = self.aggregator(x)  # [B, num_granularities]

        # Apply noise and dropout ONLY during training
        if self.training:
            # A. Noise Injection
            if self.agg_noise_std > 0:
                # Log-space approximation: log(p) + noise -> softmax
                logits_approx = torch.log(scale_weights + 1e-6) + torch.randn_like(scale_weights) * self.agg_noise_std
                scale_weights = F.softmax(logits_approx, dim=-1)

            # B. Scale Dropout
            if self.scale_dropout > 0:
                keep_prob = 1.0 - self.scale_dropout
                mask = torch.bernoulli(torch.full_like(scale_weights, keep_prob))
                
                # Safety: Ensure we don't drop ALL scales for any sample
                mask_sum = mask.sum(dim=-1)
                all_zero = (mask_sum == 0)
                if all_zero.any():
                    # Turn one random scale back on (simplified fix)
                    # Just force the first scale on for those samples
                    mask[all_zero, 0] = 1.0
                
                scale_weights = scale_weights * mask
                # Re-normalize
                scale_weights = scale_weights / (scale_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Teacher path (full signal -> target representations)
        h_list, y_tgt = self.forward_target(x, mask_y_list)

        # Student path (masked -> prediction + reconstruction)
        # Pass the perturbed scale_weights
        z_enc_list, z_pred_list, z_comb_list, r = self.forward_context(
            x, mask_x_list, mask_y_list, scale_weights=scale_weights
        )

        # Optional layer norm before alignment
        if self.USE_LN:
            h_list = [_maybe_ln(h, True) for h in h_list]
            z_enc_list = [_maybe_ln(z, True) for z in z_enc_list]
            z_pred_list = [_maybe_ln(z, True) for z in z_pred_list]

        # ---------------- Alignment losses (flattened version) ----------------
        loss_align_vis = torch.zeros((), device=device)
        loss_align_mask = torch.zeros((), device=device)

        if self.USE_LOSS_A and self.USE_ALIGN_VIS:
            h_vis = _gather_tokens_multiscale(h_list, mask_x_list)        # [Total_vis, D]
            z_vis = _gather_tokens_multiscale(z_enc_list, mask_x_list)    # [Total_vis, D]
            if h_vis.numel() > 0 and z_vis.numel() > 0:
                loss_align_vis = self.loss_fn(h_vis, z_vis)

        if self.USE_LOSS_A and self.USE_ALIGN_MASK:
            h_mask = _gather_tokens_multiscale(h_list, mask_y_list)         # [Total_mask, D]
            z_mask = _gather_tokens_multiscale(z_pred_list, mask_y_list)    # [Total_mask, D]
            if h_mask.numel() > 0 and z_mask.numel() > 0:
                loss_align_mask = self.loss_fn(h_mask, z_mask)

        # ---------------- Reconstruction loss ----------------
        loss_recon = self.loss_fn(y_tgt, r) if self.USE_LOSS_R else torch.zeros((), device=device)

        # ---------------- [NEW] Entropy Regularization ----------------
        loss_entropy = torch.zeros((), device=device)
        if self.training and self.entropy_lambda > 0:
            # We want to encourage higher entropy (prevent collapse to single scale)
            # Loss = - lambda * H(w) = lambda * sum(p log p)
            w_safe = scale_weights + 1e-6
            entropy = -torch.sum(w_safe * torch.log(w_safe), dim=-1).mean()
            loss_entropy = -self.entropy_lambda * entropy

        # Get effective loss weights (schedule)
        eff = self._effective_loss_weights()

        # Weighted losses
        w_recon = eff["lambda_recon"] * loss_recon
        w_align_vis = eff["lambda_align_vis"] * loss_align_vis
        w_align_mask = eff["lambda_align_mask"] * loss_align_mask
        w_align_total = w_align_vis + w_align_mask

        # Total loss
        loss = torch.zeros((), device=device)
        if self.USE_LOSS_A and self.USE_LOSS_R:
            loss = w_recon + w_align_total
        elif self.USE_LOSS_A:
            loss = w_align_total
        elif self.USE_LOSS_R:
            loss = w_recon
            
        # Add entropy loss
        loss = loss + loss_entropy

        # Logging
        logs = {
            "align_vis": float(loss_align_vis.detach()),
            "align_mask": float(loss_align_mask.detach()),
            "recon": float(loss_recon.detach()),
            "entropy_loss": float(loss_entropy.detach()),
            "w_recon_factor": float(eff["lambda_recon"]),
            "w_align_vis_factor": float(eff["lambda_align_vis"]),
            "w_align_mask_factor": float(eff["lambda_align_mask"]),
            "recon_w": float(w_recon.detach()),
            "align_vis_w": float(w_align_vis.detach()),
            "align_mask_w": float(w_align_mask.detach()),
            "align_w": float(w_align_total.detach()),
        }

        # Scale weights per granularity
        for i in range(self.num_granularities):
            logs[f"scale_weight_{i}"] = float(scale_weights[:, i].mean().cpu().item())

        # Optional frequency stats
        if "magnitude" in freq_info:
            logs["freq_magnitude_avg"] = float(freq_info["magnitude"].mean().cpu().item())

        # ---------------------------------------------------------
        # Monitor Logic (ALL Scales)
        # ---------------------------------------------------------
        # Only run check periodically to save compute
        if (self.global_step % 100 == 0) and (len(h_list) > 0):
            
            # Loop through all scales
            for i, h_scale in enumerate(h_list):
                p_size = self.patch_sizes[i]
                
                # h_scale shape: [B, C, N, D]
                # Monitor expects: [B, Time, Channel, Dim]
                h_monitor = h_scale.permute(0, 2, 1, 3) 
                
                # Compute stats
                stats = self.monitor.check(h_monitor)
                
                # Rename keys to include patch size
                # e.g. "collapse/eff_rank" -> "collapse_p200/eff_rank"
                for k, v in stats.items():
                    category, metric = k.split('/') # e.g. "collapse", "eff_rank"
                    new_key = f"{category}_p{p_size}/{metric}"
                    logs[new_key] = v
        # ---------------------------------------------------------

        return loss, loss_recon, (loss_align_vis + loss_align_mask), logs
    
    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x = batch
        loss, loss_recon, loss_align, logs = self._compute_losses(x)
        # print("loss: ", loss)
        # print("loss recon: ", loss_recon)
        # print("loss align: ", loss_align)
        # Log metrics
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_loss_recon", loss_recon, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_loss_align", loss_align, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_align_vis", logs["align_vis"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_align_mask", logs["align_mask"], on_epoch=True, on_step=False, sync_dist=True)
        
        self.log("w_recon", logs["w_recon_factor"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("w_align_vis", logs["w_align_vis_factor"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("w_align_mask", logs["w_align_mask_factor"], on_epoch=True, on_step=False, sync_dist=True)
        
        # Log scale weights
        for i in range(self.num_granularities):
            self.log(f"train_scale_weight_{i}", logs[f"scale_weight_{i}"], on_epoch=True, on_step=False, sync_dist=True)
        
        # Log collapse metrics
        for k, v in logs.items():
            if k.startswith("collapse") or k.startswith("debug"):
                self.log(k, v, on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        loss, loss_recon, loss_align, logs = self._compute_losses(x)
        
        # Log metrics
        self.log("valid_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss_recon", loss_recon, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_loss_align", loss_align, on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_align_vis", logs["align_vis"], on_epoch=True, on_step=False, sync_dist=True)
        self.log("valid_align_mask", logs["align_mask"], on_epoch=True, on_step=False, sync_dist=True)
        
        # Log scale weights
        for i in range(self.num_granularities):
            self.log(f"valid_scale_weight_{i}", logs[f"scale_weight_{i}"], on_epoch=True, on_step=False, sync_dist=True)
        
        return loss
    
    def on_train_batch_start(self, batch: Any, batch_idx: int):
        self.target_encoder.eval()
        if self.wd_scheduler is not None:
            self.wd_scheduler.step()
        return super().on_train_batch_start(batch, batch_idx)
    
    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # Update EMA teacher
        if self.momentum_scheduler is not None:
            with torch.no_grad():
                m = next(self.momentum_scheduler)
                for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                    param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)
        
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        # Separate param groups (no WD on bias/norm)
        param_groups = [
            # Encoder weights
            {"params": (p for n, p in self.encoder.named_parameters() 
                        if ("bias" not in n) and (len(p.shape) != 1))},
            # Predictor weights
            {"params": (p for n, p in self.predictor.named_parameters() 
                        if ("bias" not in n) and (len(p.shape) != 1))},
            # Aggregator weights
            {"params": (p for n, p in self.aggregator.named_parameters() 
                        if ("bias" not in n) and (len(p.shape) != 1))},
            # Decoder weights
            {"params": (p for n, p in self.decoder.named_parameters() 
                        if ("bias" not in n) and (len(p.shape) != 1))},
            # Patch embed weights
            {"params": (p for n, p in self.patch_embed.named_parameters() 
                        if ("bias" not in n) and (len(p.shape) != 1))},
            # All biases/norms (no WD)
            {"params": (p for n, p in self.named_parameters() 
                        if ("bias" in n) or (len(p.shape) == 1)),
             "weight_decay": 0.0},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, lr=self.pretrain_max_lr)
        
        total_steps = int(self.max_epochs * self.steps_per_epoch)
        warmup_steps = max(500, int(0.1 * total_steps))
        
        # LR schedule: warmup + cosine
        warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0.0)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
        
        lr_dict = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "valid_loss",
            "strict": True,
            "name": None,
        }
        
        # Weight decay schedule
        self.wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=self.ref_wd,
            final_wd=self.final_wd,
            T_max=total_steps,
        )
        
        # EMA momentum schedule
        m0, m1 = self.ema_start, self.ema_end
        
        def ema_cosine():
            for i in range(total_steps + 1):
                yield m1 - (m1 - m0) * 0.5 * (1 + math.cos(math.pi * i / total_steps))
        
        self.momentum_scheduler = ema_cosine()
        
        return {"optimizer": optimizer, "lr_scheduler": lr_dict}