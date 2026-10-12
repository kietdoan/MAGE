from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
from .utils import LinearWithConstraint

# ---------- shape helper ----------
def _as_BNED(z: torch.Tensor) -> torch.Tensor:
    """
    Normalize inputs to [B, N, E, D]:
      - [B, D]       -> [B, 1, 1, D]
      - [B, N, D]    -> [B, N, 1, D]
      - [B, N, E, D] -> as-is
    """
    if z.dim() == 2:  # [B, D]
        return z[:, None, None, :]
    if z.dim() == 3:  # [B, N, D]
        return z[:, :, None, :]
    if z.dim() == 4:
        return z
    raise ValueError(f"Expected 2/3/4D tensor, got {tuple(z.shape)}")

# ---------- 1) Average pooling -> MLP ----------
class AvgPoolMLPHead(nn.Module):
    """ Global mean over (N,E), then MLP. """
    def __init__(self, *, embed_dim: int, num_classes: int,
                 hidden: int = 128, dropout: float = 0.5,
                 max_norm_fc1: float = 1.0, max_norm_fc2: float = 0.25):
        super().__init__()
        self.D = int(embed_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)
        self.fc1 = LinearWithConstraint(self.D, hidden, max_norm=max_norm_fc1)
        self.fc2 = LinearWithConstraint(hidden, num_classes, max_norm=max_norm_fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)                # [B,N,E,D]
        x = z.mean(dim=(1, 2))         # [B,D]
        h = self.act(self.fc1(self.drop(x)))
        return self.fc2(h)

class EEGPTProbeHead(nn.Module):
    """
    Match LitEEGPTCausal probe:
      z [B,N,E,D]
        -> flatten last two dims per time: [B,N,E*D]
        -> Dropout
        -> Linear(E*D -> P) per time (no nonlinearity)
        -> Flatten across time: [B, N*P]
        -> Linear(N*P -> num_classes)
    """
    def __init__(self, *,
                 n_time: int, embed_slots: int, embed_dim: int,
                 num_classes: int, proj_dim: int = 16,
                 dropout: float = 0.5,
                 max_norm_fc1: float = 1.0, max_norm_fc2: float = 0.25):
        super().__init__()
        self.N = int(n_time)
        self.E = int(embed_slots)
        self.D = int(embed_dim)
        self.P = int(proj_dim)

        self.drop = nn.Dropout(float(dropout))
        self.proj = LinearWithConstraint(self.E * self.D, self.P, max_norm=max_norm_fc1)
        self.cls  = LinearWithConstraint(self.N * self.P, num_classes, max_norm=max_norm_fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = _as_BNED(z)                 # [B,N,E,D]
        B, N, E, D = z.shape
        if (N, E, D) != (self.N, self.E, self.D):
            raise ValueError(
                f"EEGPTProbeHead expects (N,E,D)=({self.N},{self.E},{self.D}), got ({N},{E},{D})."
            )
        h = z.reshape(B, N, E * D)      # [B,N,E*D]
        h = self.proj(self.drop(h))     # [B,N,P]
        h = h.reshape(B, N * self.P)    # [B,N*P]
        return self.cls(h)              # [B,K]

# ---------- factory ----------
def make_head(
    head_type: str,
    *,
    num_classes: int,
    embed_dim: Optional[int] = None,
    hidden: int = 256,
    dropout: float = 0.5,
    n_time: Optional[int] = None, 
    embed_slots: Optional[int] = None, 
    proj_dim: int = 16,    
) -> nn.Module:
    ht = head_type.lower()

    if ht in {"avg", "average", "gap"}:
        assert embed_dim is not None, "AvgPoolMLPHead requires embed_dim"
        return AvgPoolMLPHead(embed_dim=int(embed_dim),
                              num_classes=int(num_classes),
                              hidden=int(hidden),
                              dropout=float(dropout))
    if ht == "eegpt_probe":
        assert (n_time is not None) and (embed_slots is not None) and (embed_dim is not None), \
            "EEGPTProbeHead requires n_time, embed_slots, embed_dim"
        return EEGPTProbeHead(n_time=int(n_time),
                              embed_slots=int(embed_slots),
                              embed_dim=int(embed_dim),
                              num_classes=int(num_classes),
                              proj_dim=int(proj_dim),
                              dropout=float(dropout))
    raise ValueError("Unknown head_type. Choose one of: "
                     "'avg', 'eegpt_probe'")