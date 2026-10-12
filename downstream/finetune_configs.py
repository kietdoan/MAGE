"""
Minimal defaults for downstream fine-tuning (encoder-only).

- No dataset loaders, no pretrain selection, no predictor configs.
- Runner reads these constants as CLI defaults (override via flags).
- Architecture presets expose ONLY the encoder block needed by the modules.
"""

from __future__ import annotations
from typing import Dict, Optional, List

# ---------------------------
# Repro / run scale
# ---------------------------
SEED: int = 7
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 100

# Devices default (override via --devices)
devices: List[int] = [0]

MAX_LR_HEAD: float = 4e-4
MAX_LR_ENC: float = 1e-4
WD_HEAD: float = 0.01
WD_ENC: float = 0.05

# Scheduler
SCHEDULER: str = "onecycle"
WARMUP_PCT: float = 0.20

tag: str = "custom_64d_6L"
variant: str = "D" 
preprocess: str = "rep" 
encoder: str = "full_token" 
transformer: str = "crisscross" 

# ---------------------------
# Frontend / head defaults (downstream)
# ---------------------------
IN_CHANS: int = 22
TARGET_CHANS: int = 19
SEQ_LEN: int = 1024
PATCH: int = 64
DROPOUT: float = 0.5

HEAD_TYPE: str = "eegpt_probe"
POOL: str = "mean"

USE_DYNAMIC_PE: bool = False
ROPE_TIME_DEFAULT: bool = True
ROPE_CHAN_DEFAULT: bool = False
ROPE_DIM_FRAC_DEFAULT: float = 1.0
ROPE_INTERP_TIME_DEFAULT: float = 1.0
ROPE_INTERP_CHAN_DEFAULT: float = 1.0
PHYSIOMI_TEST_FOLD_INDEX: int = 4
PHYSIOMI_CV_FOLD_INDEX: Optional[int] = None

PHYSIOMI_FOLDS_JSON: Optional[str] = None

def get_config(tag_name: Optional[str] = None) -> Dict[str, Dict]:
    name = tag_name or tag
    cfg = MODELS_CONFIGS.get(name)
    if cfg is None:
        raise KeyError(f"Unknown tag '{name}'. Available: {list(MODELS_CONFIGS)}")
    return {"encoder": cfg["encoder"]}

MAGE_CONFIGS: Dict[str, Dict] = {
    "custom_64d_6L": {  # Or any name you prefer
        "patch_sizes": [50, 100, 200],
        "k_temporal": 3,
        "temporal_extra_convs": 2,
        "use_spectral": True,
        "aggregator_hidden_dim": 128,
        "use_frequency": True,
        "aggregator_freq_dim": 64,
    },
    "mage_128d_8L": {
        "patch_sizes": [50, 100, 200],
        "k_temporal": 3,
        "temporal_extra_convs": 2,
        "use_spectral": True,
        "aggregator_hidden_dim": 32,
        "use_frequency": True,
        "aggregator_freq_dim": 16,
    },
}

# Encoder configurations for MAGE
MODELS_CONFIGS: Dict[str, Dict] = {
    "custom_64d_6L": {"encoder": {"embed_dim": 64, "num_heads": 4, "depth": 6}},
    "mage_128d_8L": {
        "encoder": {
            "embed_dim": 128,
            "num_heads": 4, 
            "depth": 8
        }
    },
}

def get_mage_configs(tag_name: Optional[str] = None) -> Dict:
    """
    Return config for downstream fine-tuning.
    """
    name = tag_name or tag
    cfg = MAGE_CONFIGS.get(name)
    if cfg is None:
        print(f"[WARN] No mage config for '{name}', using default")
        return {
            "patch_sizes": [50, 100, 200],
            "k_temporal": 3,
            "temporal_extra_convs": 2,
            "use_spectral": True,
            "aggregator_hidden_dim": 128,
            "use_frequency": True,
            "aggregator_freq_dim": 64,
        }
    return cfg

def get_full_config(tag_name: Optional[str] = None) -> Dict[str, Dict]:
    name = tag_name or tag
    base = get_config(name)
    
    ms = get_mage_configs(name)
    return {
        "encoder": base["encoder"],
        "multiscale": ms,
    }

__all__ = [
    "SEED", "BATCH_SIZE", "MAX_EPOCHS", "devices",
    "MAX_LR_HEAD", "MAX_LR_ENC", "WD_HEAD", "WD_ENC",
    "SCHEDULER", "WARMUP_PCT",
    "tag", "variant", "preprocess", "encoder", "transformer",
    "IN_CHANS", "TARGET_CHANS", "SEQ_LEN", "PATCH", "DROPOUT",
    "HEAD_TYPE", "POOL",
    "USE_DYNAMIC_PE", "ROPE_TIME_DEFAULT", "ROPE_CHAN_DEFAULT",
    "ROPE_DIM_FRAC_DEFAULT", "ROPE_INTERP_TIME_DEFAULT", "ROPE_INTERP_CHAN_DEFAULT",
    "MODELS_CONFIGS", "get_config",
    "PHYSIOMI_TEST_FOLD_INDEX", "PHYSIOMI_CV_FOLD_INDEX", "PHYSIOMI_FOLDS_JSON",
    "get_full_config",
    "MAGE_CONFIGS", "get_mage_configs",
]