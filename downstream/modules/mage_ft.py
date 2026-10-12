from __future__ import annotations
from typing import Sequence, Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  # Added for metric handling

from .utils import Conv1dWithConstraint
from .base import LitBase
from .heads import make_head

try:
    from .optim import build_optim_and_sched  # optional
except Exception:
    build_optim_and_sched = None  # type: ignore

from downstream.backbones.mage import (
    MultiScalePatchEmbedding,
    ChannelTimeMAGEEncoder,
    MAGEAggregator,
)

# Metrics (tolerant import)
try:
    from utils_eval import get_metrics
except Exception:
    try:
        from downstream.utils_eval import get_metrics  # type: ignore
    except Exception:
        get_metrics = None


# ---- Default montages ----
DEFAULT_CH_58: Sequence[str] = (
    "FP1","FPZ","FP2","AF3","AF4","F7","F5","F3","F1","FZ","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCZ","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","CZ","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPZ","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","PZ","P2","P4","P6","P8",
    "PO7","PO3","POZ","PO4","PO8","O1","OZ","O2",
)
DEFAULT_CH_19: Sequence[str] = (
    "FP1","FP2","F7","F3","FZ","F4","F8",
    "T7","C3","CZ","C4","T8",
    "P7","P3","PZ","P4","P8",
    "O1","O2",
)


class LitMAGEDownstream(LitBase):
    """
    MAGE fine-tuning for downstream tasks.
    
    Architecture:
    1. Optional 1x1 channel-mixing conv
    2. Multi-scale patch embedding (frozen)
    3. Multi-scale encoder (frozen/partially frozen)
    4. Per-scale classification heads (trainable)
    5. Aggregator learns to weight scale predictions (trainable)
    6. Final weighted prediction
    """

    def __init__(
        self,
        load_path: Optional[str],
        *,
        # Dataset/model channels
        in_chans: int = 22,
        channel_names: Optional[Sequence[str]] = None,
        target_chans: int = 19,
        use_channel_names: Optional[Sequence[str]] = None,
        # Sequence geometry
        time_length: int = 6000,
        patch_sizes: List[int] = [200, 300, 600],
        # Model config
        d_model: int = 256,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        dropout: float = 0.1,
        no_inter_attn_enc: bool = False,
        use_channel_attn: bool = True,
        use_rope_time: bool = True,
        # Patch embedding config
        k_temporal: int = 3,
        temporal_extra_convs: int = 2,
        use_spectral: bool = True,
        # Aggregation config
        aggregator_hidden_dim: int = 32,
        use_frequency: bool = True,
        aggregator_freq_dim: int = 16,
        # Head / task
        head_type: str = "avg",  # avg, cls, attention
        num_classes: int = 4,
        # Front-end toggle
        use_chan_conv: bool = True,
        chan_max_norm: float = 1.0,
        # Training policy
        freeze_encoder: bool = True,
        freeze_patch_embed: bool = True,
        freeze_aggregator: bool = True,
        ft_top_k: int = 0,
        # Optim/schedule
        lr_head: float = 4e-4,
        lr_enc: float = 1e-4,
        wd_head: float = 0.01,
        wd_enc: float = 0.05,
        max_epochs: int = 100,
        steps_per_epoch: int = 100,
        scheduler: str = "onecycle",
        warmup_pct: float = 0.2,
        # Loss tweaks
        label_smoothing: float = 0.0,
        class_weights: Optional[Sequence[float]] = None,
        
        scale_dropout: float = 0.0,
        agg_noise_std: float = 0.0,
        entropy_lambda: float = 0.0,
        active_scale: Optional[int] = None,
        # MAGE scale selection: -1 = ensemble (all heads + aggregator), >=0 = single scale.
        active_head_index: int = -1,
        
        load_pretrained: bool = True,
        init_scheme: str = "xavier_uniform",
    ):
        super().__init__()

        # --- Loss config ---
        self._ls_eps = float(label_smoothing)
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w, persistent=False)
        else:
            self.class_weights = None

        # --- Anti-overfitting Config ---
        self.scale_dropout = float(scale_dropout)
        self.initial_agg_noise_std = float(agg_noise_std)
        self.agg_noise_std = float(agg_noise_std)
        self.entropy_lambda = float(entropy_lambda)
        
        self.last_scale_weights = None

        # --- Channels / montage ---
        if channel_names is not None:
            dataset_ch_names = list(channel_names)
            in_ch_for_adapter = len(dataset_ch_names)
        else:
            dataset_ch_names = None
            in_ch_for_adapter = int(in_chans)
        
        print(f"[INFO] dataset_ch_names = {dataset_ch_names}")
        print(f"[INFO] in_ch_for_adapter = {in_ch_for_adapter}")
        
        if use_channel_names is None:
            use_channel_names = list(DEFAULT_CH_19 if (target_chans == 19) else DEFAULT_CH_58)
        else:
            use_channel_names = list(use_channel_names)

        computed_target = len(use_channel_names)
        if target_chans is None:
            target_chans = computed_target
        elif target_chans != computed_target:
            raise ValueError(f"target_chans ({target_chans}) != len(use_channel_names) ({computed_target})")

        self._dataset_channel_names = dataset_ch_names
        self.use_channel_names = use_channel_names
        self.in_chans = int(in_ch_for_adapter)
        self.target_chans = int(target_chans)

        # --- Geometry ---
        self.time_length = int(time_length)
        self.patch_sizes = patch_sizes
        self.num_granularities = len(patch_sizes)
        # MAGE selection: active_head_index -1 == ensemble, >=0 == single scale.
        # Map active_head_index onto active_scale when the latter is not explicitly set.
        if active_scale is None and active_head_index is not None and active_head_index >= 0:
            active_scale = active_head_index
        self.active_head_index = int(active_head_index if active_head_index is not None else -1)
        self.is_ensemble = (active_scale is None)
        self.active_scale = active_scale
        if active_scale is not None and not 0 <= active_scale < self.num_granularities:
            raise ValueError(f"active_scale must index patch_sizes (0..{self.num_granularities - 1})")
        self.d_model = int(d_model)
        
        # --- Task Type Setup (Added is_binary logic) ---
        self.num_classes = int(num_classes)
        self.is_binary = False
        if self.num_classes == 2:
            self.is_binary = True
        print(f"[MAGE] Num classes: {self.num_classes} | Is Binary: {self.is_binary}")

        # Validate patch sizes
        for p in patch_sizes:
            if time_length % p != 0:
                raise ValueError(f"time_length {time_length} must be divisible by patch_size {p}")

        self.example_input_array = torch.zeros(8, self.in_chans, self.time_length, dtype=torch.float32)

        # --- Optional 1x1 channel-mixing conv ---
        self.use_chan_conv = bool(use_chan_conv)
        self.chan_conv = (
            Conv1dWithConstraint(self.in_chans, self.target_chans, kernel_size=1, max_norm=chan_max_norm)
            if self.use_chan_conv else None
        )
        self.chan_norm = None

        # --- Multi-scale patch embedding ---
        self.patch_embed = MultiScalePatchEmbedding(
            time_length=time_length,
            num_channels=self.target_chans,
            patch_sizes=patch_sizes,
            d_model=d_model,
            k_temporal=k_temporal,
            temporal_extra_convs=temporal_extra_convs,
            dropout=dropout,
            use_spectral=use_spectral,
        )

        # --- Multi-scale encoder ---
        self.encoder = ChannelTimeMAGEEncoder(
            num_granularities=self.num_granularities,
            depth=encoder_depth,
            d_model=d_model,
            n_heads=encoder_heads,
            d_ff=d_model * 4,
            dropout=dropout,
            output_attention=False,
            no_inter_attn=no_inter_attn_enc,
            use_channel_attn=use_channel_attn,
            use_rope_time=use_rope_time,
        )

        # --- Per-scale classification heads ---
        head_patch_sizes = ([patch_sizes[active_scale]] if active_scale is not None else patch_sizes)
        self.scale_heads = nn.ModuleList([
            make_head(
                head_type=head_type,
                num_classes=num_classes,
                n_time=time_length // p,
                embed_slots=self.target_chans,
                embed_dim=d_model,
                dropout=dropout,
            )
            for p in head_patch_sizes
        ])

        # --- Aggregator for scale weighting ---
        self.aggregator = None if active_scale is not None else MAGEAggregator(
            time_length=time_length,
            num_channels=self.target_chans,
            patch_sizes=patch_sizes,
            hidden_dim=aggregator_hidden_dim,
            use_frequency=use_frequency,
            freq_dim=aggregator_freq_dim,
            act="gelu",
        )

        # --- Load pretrained weights ---
        self._loaded_pretrain = False
        if load_pretrained and load_path:
            try:
                pre = torch.load(load_path, map_location="cpu")
                state = pre.get("state_dict", pre)
                
                # Load patch embedding
                patch_state = {k[len("patch_embed."):]: v for k, v in state.items()
                               if isinstance(k, str) and k.startswith("patch_embed.")}
                if patch_state:
                    target = self.patch_embed.state_dict()
                    patch_state = {k: v for k, v in patch_state.items()
                                   if k in target and target[k].shape == v.shape}
                    missing, unexpected = self.patch_embed.load_state_dict(patch_state, strict=False)
                    if self.global_rank == 0:
                        print(f"[PATCH EMBED LOAD] matched {len(patch_state) - len(missing)}/{len(patch_state)}")
                
                # Load encoder
                enc_state = {k[len("encoder."):]: v for k, v in state.items()
                             if isinstance(k, str) and k.startswith("encoder.")}
                if enc_state:
                    target = self.encoder.state_dict()
                    enc_state = {k: v for k, v in enc_state.items()
                                 if k in target and target[k].shape == v.shape}
                    missing, unexpected = self.encoder.load_state_dict(enc_state, strict=False)
                    matched = len(enc_state) - len(missing)
                    if self.global_rank == 0:
                        print(f"[ENCODER LOAD] matched {matched}/{len(enc_state)} (missing={len(missing)}, unexpected={len(unexpected)})")
                
                # Load aggregator
                agg_state = {k[len("aggregator."):]: v for k, v in state.items()
                             if isinstance(k, str) and k.startswith("aggregator.")}
                if agg_state and self.aggregator is not None:
                    target = self.aggregator.state_dict()
                    agg_state = {k: v for k, v in agg_state.items()
                                 if k in target and target[k].shape == v.shape}
                    missing, unexpected = self.aggregator.load_state_dict(agg_state, strict=False)
                    if self.global_rank == 0:
                        print(f"[AGGREGATOR LOAD] matched {len(agg_state) - len(missing)}/{len(agg_state)}")
                
                self._loaded_pretrain = True
            except Exception as e:
                if self.global_rank == 0:
                    print(f"[WARN] Failed to load weights from {load_path}: {e}")

        if not self._loaded_pretrain:
            if self.global_rank == 0:
                print(f"[INIT] Randomly initializing (scheme={init_scheme})")
            self._reinit_all(init_scheme)

        # --- Training policy ---
        self._freeze_encoder = bool(freeze_encoder)
        self._freeze_patch_embed = False
        self._freeze_aggregator = bool(freeze_aggregator)
        self._ft_top_k = int(ft_top_k)
        self._apply_finetune_policy()

        self._max_epochs = int(max_epochs)
        self._steps_per_epoch = int(steps_per_epoch)
        self._lr_head, self._lr_enc = float(lr_head), float(lr_enc)
        self._wd_head, self._wd_enc = float(wd_head), float(wd_enc)
        self._scheduler, self._warmup_pct = str(scheduler), float(warmup_pct)

        self.running_scores = {"train": [], "valid": [], "test": []}

    # ---- Init helpers ----
    @staticmethod
    def _init_module(m: nn.Module, scheme: str = "xavier_uniform") -> None:
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
                if scheme == "xavier_uniform":
                    nn.init.xavier_uniform_(m.weight)
                elif scheme == "xavier_normal":
                    nn.init.xavier_normal_(m.weight)
                elif scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def _reinit_all(self, scheme: str = "xavier_uniform") -> None:
        self.encoder.apply(lambda m: self._init_module(m, scheme))
        self.patch_embed.apply(lambda m: self._init_module(m, scheme))
        if self.chan_conv is not None:
            self.chan_conv.apply(lambda m: self._init_module(m, scheme))
        if self.chan_norm is not None:
            self.chan_norm.apply(lambda m: self._init_module(m, scheme))
        if self.aggregator is not None:
            self.aggregator.apply(lambda m: self._init_module(m, scheme))

    # ---- Loss ----
    def _loss_fn(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._ls_eps <= 0.0:
            return F.cross_entropy(logits, targets, weight=self.class_weights)
        C = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        y_true = F.one_hot(targets, num_classes=C).float()
        y_smooth = (1.0 - self._ls_eps) * y_true + self._ls_eps / C
        loss = -(y_smooth * logp).sum(dim=-1)
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()

    # ---- Fine-tune policy ----
    def _get_blocks(self) -> Optional[List[nn.Module]]:
        if hasattr(self.encoder, "layers"):
            return list(self.encoder.layers)
        return None
    def print_layer_training_status(self):
        """
        Print the training status (frozen/trainable) of each component before training.
        Call this after model initialization and before training starts.
        """
        print("\n" + "="*80)
        print("LAYER TRAINING STATUS")
        print("="*80)
        
        def count_params(module):
            """Count total and trainable parameters in a module."""
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable
        
        def print_module_status(name, module):
            """Print status of a module."""
            total, trainable = count_params(module)
            status = "TRAINABLE" if trainable > 0 else "FROZEN"
            pct = (trainable / total * 100) if total > 0 else 0
            print(f"{name:40s} | {status:10s} | {trainable:>10,} / {total:>10,} ({pct:>5.1f}%)")
            return total, trainable
        
        total_params = 0
        total_trainable = 0
        
        # 1. Channel Conv
        print("\n1. CHANNEL MIXING CONV (1x1)")
        print("-" * 80)
        if self.chan_conv is not None:
            t, tr = print_module_status("  chan_conv", self.chan_conv)
            total_params += t
            total_trainable += tr
        else:
            print("  Not used")
        if self.chan_norm is not None:
            t, tr = print_module_status("  chan_norm", self.chan_norm)
            total_params += t
            total_trainable += tr
        
        # 2. Multi-scale Patch Embedding
        print("\n2. MULTI-SCALE PATCH EMBEDDING")
        print("-" * 80)
        t, tr = print_module_status("  patch_embed (total)", self.patch_embed)
        total_params += t
        total_trainable += tr
        
        if hasattr(self.patch_embed, 'embeddings'):
            for i, emb in enumerate(self.patch_embed.embeddings):
                print_module_status(f"    scale_{i} (patch_size={self.patch_sizes[i]})", emb)
        
        # 3. Multi-scale Encoder
        print("\n3. MULTI-SCALE ENCODER")
        print("-" * 80)
        t, tr = print_module_status("  encoder (total)", self.encoder)
        total_params += t
        total_trainable += tr
        
        # Print per-layer status if available
        blocks = self._get_blocks()
        if blocks is not None:
            for i, block in enumerate(blocks):
                t_blk, tr_blk = count_params(block)
                status = "TRAINABLE" if tr_blk > 0 else "FROZEN"
                print(f"    layer_{i:02d}                                        | {status:10s} | {tr_blk:>10,} / {t_blk:>10,}")
        
        # Final norm
        if hasattr(self.encoder, 'norm'):
            print_module_status("    final_norm", self.encoder.norm)
        
        # 4. Per-scale Classification Heads
        print("\n4. PER-SCALE CLASSIFICATION HEADS")
        print("-" * 80)
        for i, head in enumerate(self.scale_heads):
            scale_i = self.active_scale if self.active_scale is not None else i
            t, tr = print_module_status(f"  scale_{scale_i}_head (patch_size={self.patch_sizes[scale_i]})", head)
            total_params += t
            total_trainable += tr
        
        # 5. Multi-scale Aggregator
        print("\n5. MULTI-SCALE AGGREGATOR")
        print("-" * 80)
        if self.aggregator is not None:
            t, tr = print_module_status("  aggregator", self.aggregator)
            total_params += t
            total_trainable += tr
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total parameters:     {total_params:>15,}")
        print(f"Trainable parameters: {total_trainable:>15,}")
        print(f"Frozen parameters:    {total_params - total_trainable:>15,}")
        print(f"Trainable percentage: {total_trainable/total_params*100:>14.2f}%")
        print("="*80)
        
        # Training policy summary
        print("\nTRAINING POLICY:")
        print(f"  - Freeze encoder: {self._freeze_encoder}")
        print(f"  - Freeze patch embedding: {self._freeze_patch_embed}")
        print(f"  - Fine-tune top-k encoder layers: {self._ft_top_k}")
        print(f"  - Use channel conv: {self.use_chan_conv}")
        print(f"  - Loaded pretrained weights: {self._loaded_pretrain}")
        print("="*80 + "\n")

    def _apply_finetune_policy(self) -> None:
        """
        Apply fine-tuning policy:
        - Encoder: frozen or partially unfrozen
        - Patch embedding: frozen
        - Channel conv: always trainable
        - Scale heads: always trainable
        - Aggregator: always trainable
        """
        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        
        # Freeze patch embedding
        if self._freeze_patch_embed:
            for p in self.patch_embed.parameters():
                p.requires_grad_(False)
        else:
            for p in self.patch_embed.parameters():
                p.requires_grad_(True)
        
        # Channel conv always trainable
        if self.chan_conv is not None:
            for p in self.chan_conv.parameters():
                p.requires_grad_(True)
        if self.chan_norm is not None:
            for p in self.chan_norm.parameters():
                p.requires_grad_(True)
        
        for head in self.scale_heads:
            for p in head.parameters():
                p.requires_grad_(True)
        
        if self.aggregator is None:
            pass
        elif self._freeze_aggregator:
            for p in self.aggregator.parameters():
                p.requires_grad_(False)
            self.aggregator.eval()
        else:
            for p in self.aggregator.parameters():
                p.requires_grad_(True)

        if self._freeze_encoder:
            self.encoder.eval()
            return

        if self._ft_top_k <= 0:
            for p in self.encoder.parameters():
                p.requires_grad_(True)
            self.encoder.train()
            return

        blocks = self._get_blocks()
        if blocks:
            k = min(self._ft_top_k, len(blocks))
            for blk in blocks[-k:]:
                for p in blk.parameters():
                    p.requires_grad_(True)

        if hasattr(self.encoder, "norm"):
            for p in self.encoder.norm.parameters():
                p.requires_grad_(True)

        self.encoder.train()

    # ---- Forward ----
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, in_chans, T] input EEG
        Returns:
            x: original input (for logging)
            logits: [B, num_classes] classification logits
        """
        # Set encoder mode
        if any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.train()
        else:
            self.encoder.eval()
        
        xa = x  # [B, C, T]
        
        if self.use_chan_conv and self.chan_conv is not None:
            xa = self.chan_conv(xa) 
        
        x_list = self.patch_embed(xa) 

        z_list, _ = self.encoder(x_list, mask_list=None) 

        if self.active_scale is not None:
            z = z_list[self.active_scale].permute(0, 2, 1, 3)
            return x, self.scale_heads[0](z)
        
        scale_logits = []
        for z, head in zip(z_list, self.scale_heads):
            z = z.permute(0, 2, 1, 3)
            logits_i = head(z)  # [B, num_classes]
            scale_logits.append(logits_i)
        
        # Stack scale logits: [G, B, K]
        scale_logits = torch.stack(scale_logits, dim=0) 
        
        scale_weights, _ = self.aggregator(xa)  # [B, G]
        
        if self.training and self.agg_noise_std > 0:
            logits_approx = torch.log(scale_weights + 1e-6) + torch.randn_like(scale_weights) * self.agg_noise_std
            scale_weights = F.softmax(logits_approx, dim=-1)

        if self.training and self.scale_dropout > 0:
            keep_prob = 1.0 - self.scale_dropout
            mask = torch.bernoulli(torch.full_like(scale_weights, keep_prob))
            
            mask_sum = mask.sum(dim=-1) # [B]
            all_zero = (mask_sum == 0)
            if all_zero.any():
                rand_idx = torch.randint(0, self.num_granularities, (all_zero.sum(),), device=x.device)
                mask[all_zero, rand_idx] = 1.0
            
            # Apply mask
            scale_weights = scale_weights * mask
            
            # Re-normalize so weights sum to 1
            scale_weights = scale_weights / (scale_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Save for logging/loss
        self.last_scale_weights = scale_weights

        scale_weights_t = scale_weights.t().unsqueeze(-1)
        
        logits = (scale_logits * scale_weights_t).sum(dim=0)  # [B, K]
        
        return x, logits

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        hold_epochs = 20
        if self._max_epochs > hold_epochs:
            if self.current_epoch < hold_epochs:
                self.agg_noise_std = self.initial_agg_noise_std
            else:
                decay_epochs = self._max_epochs - hold_epochs
                progress = (self.current_epoch - hold_epochs) / decay_epochs
                self.agg_noise_std = self.initial_agg_noise_std * max(0.0, 1.0 - progress)

        self.log('train_agg_noise_std', self.agg_noise_std, on_epoch=True, on_step=False, sync_dist=True)

    # ---- Training loops ----
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        x_vis, logit = self.forward(x)
        
        # Main Task Loss
        loss_cls = self._loss_fn(logit, y)
        
        loss_entropy = 0.0
        if self.entropy_lambda > 0 and self.last_scale_weights is not None:
            w = self.last_scale_weights + 1e-6
            entropy = -torch.sum(w * torch.log(w), dim=-1).mean()
            loss_entropy = - self.entropy_lambda * entropy
            
        total_loss = loss_cls + loss_entropy
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log('train_loss', total_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_cls_loss', loss_cls, on_epoch=True, on_step=False, sync_dist=True)
        self.log('train_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        
        # Log Aggregator stats 
        if self.last_scale_weights is not None:
            for i in range(self.num_granularities):
                self.log(f'agg_weight/scale_{i}', self.last_scale_weights[:, i].mean(), on_epoch=True, on_step=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self._loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log('valid_loss', loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log('valid_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.running_scores["valid"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_validation_epoch_start(self) -> None:
        self.running_scores["valid"] = []
        return super().on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["valid"]) == 0:
            return super().on_validation_epoch_end()
        labels, scores = zip(*self.running_scores["valid"])
        y_true = torch.cat(list(labels), dim=0).cpu().numpy()
        y_score = torch.cat(list(scores), dim=0).cpu().numpy()
        
        # Select metrics based on task type
        if self.is_binary:
            metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1"]
        else:
            metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
            
        results = get_metrics(y_score, y_true, metrics, self.is_binary)
        
        for k, v in results.items():
            # Handle NaNs safely like in reference code
            try:
                v = float(np.nan_to_num(v, nan=0.0))
            except Exception:
                pass
            self.log('valid_' + k, torch.tensor(v, device=self.device), on_epoch=True, on_step=False, sync_dist=True)
        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.long()
        _, logit = self.forward(x)
        loss = self._loss_fn(logit, y)
        pred = logit.argmax(dim=-1)
        accuracy = (pred == y).float().mean()

        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('test_accuracy', accuracy, on_epoch=True, on_step=False, sync_dist=True)
        self.running_scores["test"].append((y.detach().cpu(), logit.detach().cpu()))
        return loss

    def on_test_epoch_start(self) -> None:
        self.running_scores["test"] = []
        return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        if get_metrics is None or len(self.running_scores["test"]) == 0:
            return super().on_test_epoch_end()
        labels, scores = zip(*self.running_scores["test"])
        y_true = torch.cat(list(labels), dim=0).cpu().numpy()
        y_score = torch.cat(list(scores), dim=0).cpu().numpy()
        
        # Select metrics based on task type
        if self.is_binary:
            metrics = ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "precision", "recall", "f1"]
        else:
            metrics = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted", "f1_macro", "f1_micro"]
            
        results = get_metrics(y_score, y_true, metrics, self.is_binary)
        
        for k, v in results.items():
            # Handle NaNs safely like in reference code
            try:
                v = float(np.nan_to_num(v, nan=0.0))
            except Exception:
                pass
            self.log('test_' + k, torch.tensor(v, device=self.device), on_epoch=True, on_step=False, sync_dist=True)
        return super().on_test_epoch_end()

    # ---- Optimizer ----
    def configure_optimizers(self):
        head_params: List[nn.Parameter] = []
        
        # Heads
        if self.scale_heads is not None:
            for head in self.scale_heads:
                head_params += list(head.parameters())
        
        # Channel mixing layers
        if self.chan_conv is not None:
            head_params += list(self.chan_conv.parameters())
        if self.chan_norm is not None:
            head_params += list(self.chan_norm.parameters())
        
        if not self._freeze_patch_embed:
            head_params += list(self.patch_embed.parameters())

        enc_params = [p for p in self.encoder.parameters() if p.requires_grad]

        if self.aggregator is not None:
            head_params += [p for p in self.aggregator.parameters() if p.requires_grad]

        groups = [{"params": head_params, "weight_decay": self._wd_head, "lr": self._lr_head}]
        
        if enc_params:
            groups.append({"params": enc_params, "weight_decay": self._wd_enc, "lr": self._lr_enc})

        optimizer = torch.optim.AdamW(groups)

        if self._scheduler.lower() == "onecycle":
            max_lrs = [self._lr_head] 
            if enc_params:
                max_lrs.append(self._lr_enc)
                
            sched = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                pct_start=self._warmup_pct,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        # Fallback to custom builder if available
        if build_optim_and_sched is not None and self._scheduler.lower() != "none":
            return build_optim_and_sched(
                groups,
                scheduler=self._scheduler,
                steps_per_epoch=self._steps_per_epoch,
                epochs=self._max_epochs,
                warmup_pct=self._warmup_pct,
            )

        return {"optimizer": optimizer}
