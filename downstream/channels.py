"""
Channel catalogs + tolerant name->index mapper.

Concepts
--------
1) CHANNELS_BY_DATASET
   - What channels exist in your *processed dataset files*.
   - Use these to decide/validate dataset-side selection and derive `in_chans`.

2) MONTAGE_BY_DATASET
   - The MODEL montage (names+order) you want the encoder to use per dataset.
   - The Lightning module’s 1x1 Conv adapts from `in_chans` -> `len(montage)`.

Utilities
---------
- make_channel_indices(available, wanted, strict=True):
    robust mapping from header names in files -> indices for slicing.

- DEFAULT_EEGPT58:
    convenient default MODEL montage (your 58-ch pretrain).
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Sequence

DEFAULT_EEGPT58 = [
    'FP1','FPZ','FP2',
    'AF7','AF3','AF4','AF8',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO5','PO3','POZ','PO4','PO6','PO8',
    'O1','OZ','O2',
]

DEFAULT_CH_19: Sequence[str] = (
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4","P8","O1","O2"
)

DEFAULT_CH_16: Sequence[str] = (
    "FP1","FP2","F7","F3","FZ","F4","F8","T7","C3","CZ","C4","T8","P7","P3","PZ","P4"
)

PHYSIOP300_62 = [
    'FP1','FPZ','FP2',
    'AF3','AF4',
    'F7','F5','F3','F1','FZ','F2','F4','F6','F8',
    'FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8',
    'T7','C5','C3','C1','CZ','C2','C4','C6','T8',
    'TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8',
    'PO7','PO3','POZ','PO4','PO8',
    'O1','OZ','O2',
]
_PHYSIOP300_DROP4 = {'PO7','PO8','AF3','AF4'}
PHYSIOP300_58 = [ch for ch in PHYSIOP300_62 if ch not in _PHYSIOP300_DROP4]

BCIC2a_22 = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'
]
BCIC2a_19 = DEFAULT_CH_19

BCIC2b_3 = ['C3','CZ','C4']
BCIC2b_MONTAGE = ['C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6']

KAGGLE_ERN_56 = "Fp1,Fp2,AF7,AF3,AF4,AF8,F7,F5,F3,F1,Fz,F2,F4,F6,F8,FT7,FC5,FC3,FC1,FCz,FC2,FC4,FC6,FT8,T7,C5,C3,C1,Cz,C2,C4,C6,T8,TP7,CP5,CP3,CP1,CPz,CP2,CP4,CP6,TP8,P7,P5,P3,P1,Pz,P2,P4,P6,P8,PO7,POz,PO8,O1,O2".split(",")
KAGGLE_ERN_56 = [x.upper() for x in KAGGLE_ERN_56]
KAGGLE_ERN_19 = DEFAULT_CH_19
KAGGLE_DATASET = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19'
]

SLEEPEDF_PAIRS = ['Fpz-Cz','Pz-Oz']
SLEEPEDF_4 = ['FPZ','CZ','PZ','OZ']
SLEEPEDF_DATASET = ['1', '2']
SLEEPEDF_EEGPT_DATASET = ['F3', 'F4']
SLEEPEDF_EEGPT = ['F3', 'F4', 'C3', 'C4', 'P3','P4', 'FPZ', 'FZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']

TUAB_19 = DEFAULT_CH_19

TUAB_DATASET = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23'
]

PHYSIOMI_64 = [
    "Fp1","Fpz","Fp2",
    "AF7","AF3","AFz","AF4","AF8",
    "F7","F5","F3","F1","Fz","F2","F4","F6","F8",
    "FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8",
    "T7","C5","C3","C1","Cz","C2","C4","C6","T8",
    "TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8",
    "P7","P5","P3","P1","Pz","P2","P4","P6","P8",
    "PO7","PO5","PO3","POz","PO4","PO6","PO8",
    "O1","Oz","O2","Iz",
]

SHUMI_32 = [
    'Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FC1', 'FC2', 'FC5', 
    'FC6', 'Cz', 'C3', 'C4', 'T3', 'T4', 
    'CP1', 'CP2', 'CP5', 'CP6', 'Pz', 'P3', 'P4', 'T5', 'T6', 
    'PO3', 'PO4', 'Oz', 'O1', 'O2',
    'A1', 'A2'
]


FACED_32 = [
    "Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", 
    "FC1", "FC2", "FC5", "FC6",
    "Cz", "C3", "C4",
    "T7", "T8",
    "CP1", "CP2", "CP5", "CP6",
    "Pz", "P3", "P4", "P7", "P8",
    "PO3", "PO4", "Oz", "O1", "O2",
    "A1", "A2"
]

CHANNELS_BY_DATASET: Dict[str, List[str]] = {
    "BCIC2a": BCIC2a_22,
    "BCIC2b": BCIC2b_3,
    "KaggleERN": KAGGLE_DATASET, 
    "PhysioP300": PHYSIOP300_58,
    "SleepEDF_4": SLEEPEDF_4,
    "TUAB": TUAB_DATASET,
    
    "KaggleERN_56": KAGGLE_ERN_56,
    "KaggleERN_19": KAGGLE_ERN_19,
    "BCIC2a_19": BCIC2a_19,
    "BCIC2b_3": BCIC2b_3,
    "PhysioP300_58": PHYSIOP300_58,
    "PhysioP300_62": PHYSIOP300_62,
    "SleepEDF_pairs": SLEEPEDF_PAIRS,
    "PhysioMI": PHYSIOMI_64,
    "SLEEPEDF": SLEEPEDF_DATASET,
    "FACED": FACED_32,
    "SHUMI": SHUMI_32,
}

MONTAGE_BY_DATASET: Dict[str, List[str]] = {
    "BCIC2a": DEFAULT_CH_19,
    "BCIC2b": BCIC2b_MONTAGE,
    "KaggleERN": DEFAULT_CH_19,
    "PhysioP300": PHYSIOP300_58,
    "TUAB": TUAB_19,           
    "PhysioMI": DEFAULT_CH_19,
    "SLEEPEDF": DEFAULT_CH_19,
    "FACED": DEFAULT_CH_19, 
    "SHUMI": DEFAULT_CH_19,
}

_ALIAS = {
    "t3": "t7", "t4": "t8", "t5": "p7", "t6": "p8",
    "fpz": "fpz", "poz": "poz", "fcz": "fcz", "cpz": "cpz",
}

def _norm(n: str) -> str:
    """
    Normalize a channel label:
      - lowercase
      - strip spaces and dots
      - strip trailing montage suffixes (-ref, -le, -re, -avg, -aver)
      - keep inner dashes (for pair labels) until comparison stage
    """
    s = n.strip().lower().replace(" ", "").replace(".", "")
    for suf in ("-ref", "-le", "-re", "-aver", "-avg"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s

def _alias(n: str) -> str:
    return _ALIAS.get(n, n)

def make_channel_indices(
    available: Iterable[str],
    wanted: Iterable[str],
    strict: bool = True
) -> List[int]:
    """
    Map `wanted` channel names to indices in `available`.

    Tolerant matching:
      - case/whitespace/dot-insensitive
      - strips EDF suffixes (-Ref, -LE, -RE, -AVG, -AVER)
      - legacy aliases (T3/T4/T5/T6 -> T7/T8/P7/P8)
      - pair labels 'Fpz-Cz': try exact pair first, else fall back to first electrode ('Fpz').

    Returns indices into `available` in the order of `wanted`.
    Raises KeyError if a wanted channel is missing and `strict=True`.
    """
    avail_raw = list(available)

    avail_norm_to_idx: Dict[str, int] = {}
    for i, a in enumerate(avail_raw):
        na = _norm(a)
        avail_norm_to_idx.setdefault(na, i)
        avail_norm_to_idx.setdefault(_alias(na.replace("-", "")), i)

    idxs: List[int] = []
    for w in wanted:
        w_norm = _norm(w)
        if "-" in w_norm and w_norm in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[w_norm])
            continue

        if "-" in w_norm:
            first = _alias(w_norm.split("-")[0]).replace("-", "")
            if first in avail_norm_to_idx:
                idxs.append(avail_norm_to_idx[first])
                continue

        key = _alias(w_norm.replace("-", ""))
        if key in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[key])
            continue

        if key.upper() in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[key.upper()]); continue
        if key.lower() in avail_norm_to_idx:
            idxs.append(avail_norm_to_idx[key.lower()]); continue

        if strict:
            raise KeyError(f"Channel '{w}' not found in available header: {available}")
    return idxs

__all__ = [
    "CHANNELS_BY_DATASET",
    "MONTAGE_BY_DATASET",
    "DEFAULT_EEGPT58",
    "make_channel_indices",
]