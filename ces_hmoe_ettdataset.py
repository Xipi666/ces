"""Causal Entropy-State Horizon-wise Mixture of Experts for ETT.

This file is an implementation-oriented reference, not a claim that the
method has already improved the ETT benchmark. It follows the standard ETT
12/4/4-month chronological split and never reads future values when building
an input window or its entropy state.

Expected CSV schema:
    date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT

Examples:
    python ces_hmoe_ettdataset.py --smoke-test
    python ces_hmoe_ettdataset.py --csv ./ETTh1.csv --dataset ETTh1
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


EPS = 1e-8


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + EPS)


def _templates(x: np.ndarray, m: int) -> np.ndarray:
    """Return overlapping vectors of shape [n-m+1, m]."""
    if len(x) < m:
        return np.empty((0, m), dtype=np.float64)
    return np.lib.stride_tricks.sliding_window_view(x, m)


def approximate_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """ApEn(m, r), including self matches as in the original definition."""
    z = _zscore(x)
    r = r_ratio
    values = []
    for order in (m, m + 1):
        vectors = _templates(z, order)
        if len(vectors) == 0:
            return 0.0
        distances = np.max(
            np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=-1
        )
        match_rate = (distances <= r).mean(axis=1)
        values.append(np.log(match_rate + EPS).mean())
    return float(np.clip(values[0] - values[1], -20.0, 20.0))


def sample_entropy(x: np.ndarray, m: int = 2, r_ratio: float = 0.2) -> float:
    """SampEn(m, r), excluding self matches.

    This quadratic implementation is used only for sparse cache anchors. The
    cheap PE and SpE features are refreshed more often in build_entropy_cache.
    """
    z = _zscore(x)
    r = r_ratio
    counts = []
    for order in (m, m + 1):
        vectors = _templates(z, order)
        if len(vectors) < 2:
            return 0.0
        distances = np.max(
            np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=-1
        )
        counts.append(float((distances <= r).sum() - len(vectors)))
    b, a = counts
    if b <= 0.0 or a <= 0.0:
        return float(-math.log((a + 1.0) / (b + 1.0)))
    return float(np.clip(-math.log((a + EPS) / (b + EPS)), 0.0, 20.0))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Normalized ordinal-pattern entropy in [0, 1]."""
    if len(x) < 1 + (order - 1) * delay:
        return 0.0
    indices = np.arange(0, order * delay, delay)
    vectors = np.asarray([x[i + indices] for i in range(len(x) - indices[-1])])
    patterns = np.argsort(vectors, axis=1, kind="stable")
    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-(probabilities * np.log(probabilities + EPS)).sum() / math.log(math.factorial(order)))


def spectral_entropy(x: np.ndarray) -> float:
    """Normalized Shannon entropy of the non-DC one-sided power spectrum."""
    z = np.asarray(x, dtype=np.float64) - np.mean(x)
    power = np.abs(np.fft.rfft(z)) ** 2
    if len(power) <= 1 or power[1:].sum() <= EPS:
        return 0.0
    probability = power[1:] / (power[1:].sum() + EPS)
    return float(-(probability * np.log(probability + EPS)).sum() / math.log(len(probability)))


@dataclass
class ETTConfig:
    csv_path: str
    dataset: str = "ETTh1"
    target: str = "OT"
    seq_len: Optional[int] = None
    pred_len: Optional[int] = None
    entropy_stride: Optional[int] = None
    expensive_entropy_stride: Optional[int] = None
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    loader_seed: Optional[int] = None

    def __post_init__(self) -> None:
        minute = self.dataset.lower().startswith("ettm")
        steps_per_hour = 4 if minute else 1
        if self.seq_len is None:
            self.seq_len = 168 * steps_per_hour
        if self.pred_len is None:
            self.pred_len = 24 * steps_per_hour
        # One day for hourly PE/SpE refresh; one hour for ETTm. Expensive
        # ApEn/SampEn are refreshed once per physical day.
        if self.entropy_stride is None:
            self.entropy_stride = 1 if not minute else 4
        if self.expensive_entropy_stride is None:
            self.expensive_entropy_stride = 24 if not minute else 96
        self.entropy_scales = (
            24 * steps_per_hour,
            48 * steps_per_hour,
            168 * steps_per_hour,
        )


class NumpyScaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std, 1e-6).astype(np.float32)

    @classmethod
    def fit(cls, x: np.ndarray) -> "NumpyScaler":
        return cls(x.mean(axis=0), x.std(axis=0))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def inverse_target(self, x: np.ndarray, target_idx: int) -> np.ndarray:
        return x * self.std[target_idx] + self.mean[target_idx]


def _calendar_features(dates: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(dates)
    hour = dt.dt.hour.to_numpy() + dt.dt.minute.to_numpy() / 60.0
    dow = dt.dt.dayofweek.to_numpy()
    month = dt.dt.month.to_numpy() - 1
    return np.stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ],
        axis=1,
    ).astype(np.float32)


def _state_one_scale(
    segment: np.ndarray,
    target_idx: int,
    ramp_threshold: float,
    include_expensive: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return [PE, SpE, mean, std, median] per channel plus target features.

    The second return value marks the two expensive target entropy slots
    (SampEn, ApEn), allowing build_entropy_cache to carry them forward.
    """
    features = []
    for channel in range(segment.shape[1]):
        x = segment[:, channel]
        features.extend(
            [
                permutation_entropy(x),
                spectral_entropy(x),
                float(np.mean(x)),
                float(np.std(x)),
                float(np.median(x)),
            ]
        )
    target = segment[:, target_idx]
    differences = np.diff(target)
    slope = float((target[-1] - target[0]) / (np.std(target) + EPS))
    ramp_ratio = float(np.mean(np.abs(differences) > ramp_threshold)) if len(differences) else 0.0
    if include_expensive:
        expensive = [sample_entropy(target), approximate_entropy(target)]
    else:
        expensive = [np.nan, np.nan]
    features.extend(expensive + [slope, ramp_ratio])
    return np.asarray(features, dtype=np.float32), np.asarray([np.isnan(expensive[0]), np.isnan(expensive[1])])


def state_dim(enc_in: int, n_scales: int) -> int:
    return n_scales * (5 * enc_in + 4)


def build_entropy_cache(
    values: np.ndarray,
    seq_len: int,
    scales: Sequence[int],
    target_idx: int,
    ramp_threshold: float,
    refresh_stride: int,
    expensive_stride: int,
) -> np.ndarray:
    """Build a causal state cache indexed by the forecast origin.

    cache[t] depends only on values[t-seq_len:t]. Cheap PE/SpE features are
    refreshed every refresh_stride origins. SampEn/ApEn are refreshed only at
    expensive_stride anchors and carried forward, which avoids a quadratic
    calculation for every ETT window while preserving causality.
    """
    n, channels = values.shape
    cache = np.zeros((n + 1, state_dim(channels, len(scales))), dtype=np.float32)
    current = cache[seq_len].copy()
    expensive_mask = []
    for _ in scales:
        expensive_mask.extend([False] * (5 * channels))
        expensive_mask.extend([True, True, False, False])
    expensive_mask = np.asarray(expensive_mask, dtype=bool)

    for origin in range(seq_len, n + 1):
        if origin == seq_len or (origin - seq_len) % refresh_stride == 0:
            chunks = []
            for scale in scales:
                segment = values[origin - min(seq_len, scale):origin]
                include_expensive = origin == seq_len or (origin - seq_len) % expensive_stride == 0
                chunk, _ = _state_one_scale(
                    segment, target_idx, ramp_threshold, include_expensive=include_expensive
                )
                chunks.append(chunk)
            fresh = np.concatenate(chunks).astype(np.float32)
            if np.any(np.isnan(fresh)):
                fresh[np.isnan(fresh)] = current[np.isnan(fresh)]
            current = fresh
        cache[origin] = current
    return cache


class ETTWindowDataset(Dataset):
    def __init__(
        self,
        values: np.ndarray,
        calendar: np.ndarray,
        entropy_cache: np.ndarray,
        border1: int,
        border2: int,
        seq_len: int,
        pred_len: int,
        target_idx: int,
    ) -> None:
        self.values = values
        self.calendar = calendar
        self.entropy_cache = entropy_cache
        self.border1 = border1
        self.border2 = border2
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx
        self.origins = list(range(border1, border2 - seq_len - pred_len + 1))

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        origin = self.origins[index] + self.seq_len
        x_start = origin - self.seq_len
        x_end = origin
        y_end = origin + self.pred_len
        x = self.values[x_start:x_end]
        y = self.values[origin:y_end, self.target_idx : self.target_idx + 1]
        future_calendar = self.calendar[origin:y_end]
        state = self.entropy_cache[origin]
        return {
            "x": torch.from_numpy(x),
            "future_calendar": torch.from_numpy(future_calendar),
            "y": torch.from_numpy(y.squeeze(-1)),
            "state": torch.from_numpy(state),
        }


class ETTDataModule:
    """Load one ETT file with the official chronological 12/4/4 split."""

    CACHE_VERSION = 1

    def __init__(self, config: ETTConfig) -> None:
        self.config = config
        csv_path = Path(config.csv_path)
        source_stat = csv_path.stat()
        cache_path = self._cache_path(csv_path)
        if self._load_cache(cache_path, source_stat):
            print(f"cache=hit path={cache_path}")
            return
        print(f"cache=miss path={cache_path}; building entropy cache")
        frame = pd.read_csv(config.csv_path)
        if "date" not in frame.columns or config.target not in frame.columns:
            raise ValueError("ETT CSV must contain date and target columns")
        frame = frame.copy()
        numeric = [c for c in frame.columns if c not in {"date", config.target}]
        numeric.append(config.target)  # Experts intentionally use OT as x[..., -1].
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        minute = config.dataset.lower().startswith("ettm")
        step = 4 if minute else 1
        num_train = 12 * 30 * 24 * step
        num_val = 4 * 30 * 24 * step
        num_test = 4 * 30 * 24 * step
        required = num_train + num_val + num_test
        if len(frame) < required:
            raise ValueError(f"Expected at least {required} rows for {config.dataset}, got {len(frame)}")
        self.borders = (num_train, num_train + num_val, required)
        # Forward fill uses only past observations; the initial gap uses a
        # mean fitted on the training prefix, never a future value.
        train_mean = frame.loc[: num_train - 1, numeric].mean()
        frame[numeric] = frame[numeric].ffill().fillna(train_mean)
        if config.target not in numeric:
            raise ValueError(f"Target {config.target!r} is not numeric")
        self.columns = numeric
        self.target_idx = numeric.index(config.target)
        raw = frame[numeric].to_numpy(np.float32)
        self.scaler = NumpyScaler.fit(raw[:num_train])
        self.values = self.scaler.transform(raw)
        self.calendar = _calendar_features(frame["date"])
        diff = np.diff(self.values[:num_train, self.target_idx])
        self.ramp_threshold = float(np.quantile(np.abs(diff), 0.90))
        self.entropy_cache = build_entropy_cache(
            self.values,
            config.seq_len,
            config.entropy_scales,
            self.target_idx,
            self.ramp_threshold,
            config.entropy_stride,
            config.expensive_entropy_stride,
        )
        self._save_cache(cache_path, source_stat)

    def _cache_path(self, csv_path: Path) -> Path:
        cache_dir = csv_path.parent / ".ett_cache"
        return cache_dir / f"{csv_path.stem}.seq{self.config.seq_len}.entropy-v{self.CACHE_VERSION}.npz"

    def _load_cache(self, cache_path: Path, source_stat: object) -> bool:
        if not cache_path.is_file():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as cached:
                if int(np.asarray(cached["cache_version"]).item()) != self.CACHE_VERSION:
                    return False
                if int(np.asarray(cached["source_size"]).item()) != source_stat.st_size:
                    return False
                if int(np.asarray(cached["source_mtime_ns"]).item()) != source_stat.st_mtime_ns:
                    return False
                if str(np.asarray(cached["dataset"]).item()) != self.config.dataset:
                    return False
                if str(np.asarray(cached["target"]).item()) != self.config.target:
                    return False
                if int(np.asarray(cached["seq_len"]).item()) != self.config.seq_len:
                    return False
                if int(np.asarray(cached["entropy_stride"]).item()) != self.config.entropy_stride:
                    return False
                if int(np.asarray(cached["expensive_entropy_stride"]).item()) != self.config.expensive_entropy_stride:
                    return False
                if not np.array_equal(cached["entropy_scales"], np.asarray(self.config.entropy_scales)):
                    return False

                self.values = cached["values"].astype(np.float32, copy=False)
                self.calendar = cached["calendar"].astype(np.float32, copy=False)
                self.entropy_cache = cached["entropy_cache"].astype(np.float32, copy=False)
                self.columns = [str(value) for value in cached["columns"].tolist()]
                self.target_idx = int(np.asarray(cached["target_idx"]).item())
                self.borders = tuple(int(value) for value in cached["borders"].tolist())
                self.ramp_threshold = float(np.asarray(cached["ramp_threshold"]).item())
                self.scaler = NumpyScaler(cached["scaler_mean"], cached["scaler_std"])
        except (OSError, KeyError, TypeError, ValueError):
            return False
        return True

    def _save_cache(self, cache_path: Path, source_stat: object) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                cache_version=np.int64(self.CACHE_VERSION),
                source_size=np.int64(source_stat.st_size),
                source_mtime_ns=np.int64(source_stat.st_mtime_ns),
                dataset=np.asarray(self.config.dataset),
                target=np.asarray(self.config.target),
                seq_len=np.int64(self.config.seq_len),
                entropy_stride=np.int64(self.config.entropy_stride),
                expensive_entropy_stride=np.int64(self.config.expensive_entropy_stride),
                entropy_scales=np.asarray(self.config.entropy_scales, dtype=np.int64),
                values=self.values,
                calendar=self.calendar,
                entropy_cache=self.entropy_cache,
                columns=np.asarray(self.columns),
                target_idx=np.int64(self.target_idx),
                borders=np.asarray(self.borders, dtype=np.int64),
                ramp_threshold=np.float32(self.ramp_threshold),
                scaler_mean=self.scaler.mean,
                scaler_std=self.scaler.std,
            )
        temp_path.replace(cache_path)
        print(f"cache=saved path={cache_path}")

    def dataset(self, split: str) -> ETTWindowDataset:
        num_train, num_val, num_total = self.borders
        if split == "train":
            border1, border2 = 0, num_train
        elif split == "val":
            border1, border2 = num_train - self.config.seq_len, num_val
        elif split == "test":
            border1, border2 = num_val - self.config.seq_len, num_total
        else:
            raise ValueError("split must be train, val, or test")
        return ETTWindowDataset(
            self.values,
            self.calendar,
            self.entropy_cache,
            border1,
            border2,
            self.config.seq_len,
            self.config.pred_len,
            self.target_idx,
        )

    def loaders(self) -> Dict[str, DataLoader]:
        train_generator = None
        if self.config.loader_seed is not None:
            train_generator = torch.Generator()
            train_generator.manual_seed(self.config.loader_seed)
        return {
            split: DataLoader(
                self.dataset(split),
                batch_size=self.config.batch_size,
                shuffle=split == "train",
                num_workers=self.config.num_workers,
                drop_last=split == "train",
                pin_memory=self.config.pin_memory,
                generator=train_generator if split == "train" else None,
            )
            for split in ("train", "val", "test")
        }


class HistoryEncoder(nn.Module):
    def __init__(self, enc_in: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(enc_in, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.transpose(1, 2)).mean(dim=-1)


class TrendExpert(nn.Module):
    """DLinear-like target-only expert: [B, L] -> [B, H]."""

    def __init__(self, seq_len: int, pred_len: int, variant: str = "v1") -> None:
        super().__init__()
        if variant not in {"v1", "v2"}:
            raise ValueError("TrendExpert variant must be 'v1' or 'v2'")
        self.variant = variant
        kernel = min(25, seq_len if seq_len % 2 else seq_len - 1)
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        self.linear = nn.Linear(seq_len, pred_len)
        self.residual_linear = nn.Linear(seq_len, pred_len) if variant == "v2" else None

    def forward(self, x: Tensor) -> Tensor:
        target = x[..., -1:].transpose(1, 2)
        trend = self.avg(target).squeeze(1)
        forecast = self.linear(trend)
        if self.residual_linear is not None:
            residual = target.squeeze(1) - trend
            forecast = forecast + 0.1 * self.residual_linear(residual)
        return forecast


class PeriodicExpert(nn.Module):
    """Small convolutional periodic expert with a direct H-vector head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        hidden: int = 32,
        variant: str = "v1",
    ) -> None:
        super().__init__()
        if variant not in {"v1", "v2", "v3"}:
            raise ValueError("PeriodicExpert variant must be 'v1', 'v2' or 'v3'")
        self.seq_len = seq_len
        self.variant = variant
        if variant == "v3":
            self.conv = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=25, padding=12),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, groups=hidden),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, groups=hidden),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(8),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=25, padding=12),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, groups=hidden),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(8),
            )
        anchor_count = 3 if variant in {"v2", "v3"} else 0
        self.anchor_head = nn.Linear(anchor_count, hidden * 2) if anchor_count else None
        head_input = hidden * 8 + (hidden * 2 if anchor_count else 0)
        self.head = nn.Linear(head_input, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        target = x[..., -1:].transpose(1, 2)
        features = [self.conv(target).flatten(1)]
        if self.anchor_head is not None:
            values = target.squeeze(1)
            anchors = []
            for lag in (24, 168, 336):
                if self.seq_len >= lag:
                    anchors.append(values[:, -lag])
                else:
                    anchors.append(values[:, 0])
            anchor_values = torch.stack(anchors, dim=-1)
            features.append(self.anchor_head(anchor_values))
        return self.head(torch.cat(features, dim=-1))


class RampExpert(nn.Module):
    """Dilated TCN expert using all historical channels and first differences."""

    def __init__(
        self,
        enc_in: int,
        pred_len: int,
        hidden: int = 48,
        variant: str = "v1",
    ) -> None:
        super().__init__()
        if variant not in {"v1", "v2", "v3"}:
            raise ValueError("RampExpert variant must be 'v1', 'v2' or 'v3'")
        self.variant = variant
        layers = [
            nn.Conv1d(enc_in, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        ]
        if variant == "v3":
            layers.extend([
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=8, dilation=4),
                nn.GELU(),
            ])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        diff = torch.diff(x, dim=1, prepend=x[:, :1])
        hidden = self.net(diff.transpose(1, 2))[:, :, -1]
        output = self.head(hidden)
        if self.variant == "v2":
            output = output + x[:, -1, -1:].expand_as(output)
        return output


class CESHMoE(nn.Module):
    """CES-HMoE; all experts return [batch, pred_len]."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        entropy_dim: int,
        calendar_dim: int = 6,
        route_dim: int = 64,
        temperature: float = 1.0,
        gate_init: str = "random",
        use_entropy_state: bool = True,
        gate_mode: str = "horizon",
        fusion_mode: str = "gate",
        prior_weights: Optional[Sequence[float]] = None,
        gate_residual_scale: float = 0.1,
        dynamic_blend: float = 0.1,
        ramp_cap: float = 0.02,
        ramp_residual_alpha: float = 0.02,
        selective_entropy: bool = False,
        entropy_adapter_scale: float = 0.01,
        trend_variant: str = "v1",
        periodic_variant: str = "v1",
        ramp_variant: str = "v1",
        gate_block_size: int = 0,
    ) -> None:
        super().__init__()
        if gate_mode not in {"horizon", "global"}:
            raise ValueError("gate_mode must be 'horizon' or 'global'")
        if fusion_mode not in {
            "gate", "concat", "static", "prior_gate", "bounded_gate", "tri_bounded_gate",
            "capped_prior_gate", "ramp_residual",
        }:
            raise ValueError(
                "fusion_mode must be 'gate', 'concat', 'static', 'prior_gate', "
                "'bounded_gate', 'tri_bounded_gate', 'capped_prior_gate' or 'ramp_residual'"
            )
        if gate_residual_scale <= 0:
            raise ValueError("gate_residual_scale must be positive")
        if not 0 < dynamic_blend <= 1:
            raise ValueError("dynamic_blend must be in (0, 1]")
        if entropy_adapter_scale <= 0:
            raise ValueError("entropy_adapter_scale must be positive")
        if not 0 < ramp_cap <= 1:
            raise ValueError("ramp_cap must be in (0, 1]")
        if not 0 <= ramp_residual_alpha <= 1:
            raise ValueError("ramp_residual_alpha must be in [0, 1]")
        self.pred_len = pred_len
        self.temperature = temperature
        self.use_entropy_state = use_entropy_state
        self.gate_mode = gate_mode
        self.fusion_mode = fusion_mode
        self.gate_residual_scale = gate_residual_scale
        self.dynamic_blend = dynamic_blend
        self.ramp_cap = ramp_cap
        self.ramp_residual_alpha = ramp_residual_alpha
        self.ramp_residual = fusion_mode == "ramp_residual"
        self.selective_entropy = selective_entropy
        self.entropy_adapter_scale = entropy_adapter_scale
        if gate_block_size < 0:
            raise ValueError("gate_block_size must be non-negative")
        self.gate_block_size = gate_block_size
        self.history = HistoryEncoder(enc_in, route_dim)
        self.state = (
            nn.Sequential(nn.LayerNorm(entropy_dim), nn.Linear(entropy_dim, route_dim), nn.GELU())
            if use_entropy_state
            else None
        )
        self.entropy_adapter = None
        if selective_entropy:
            scale_width = 5 * enc_in + 4
            if entropy_dim % scale_width != 0:
                raise ValueError("entropy_dim is incompatible with the selected entropy layout")
            target_offset = 5 * (enc_in - 1)
            selected_indices = []
            for scale_index in range(entropy_dim // scale_width):
                base = scale_index * scale_width
                selected_indices.extend(
                    [
                        base + target_offset,
                        base + target_offset + 1,
                        base + 5 * enc_in,
                        base + 5 * enc_in + 1,
                        base + 5 * enc_in + 2,
                        base + 5 * enc_in + 3,
                    ]
                )
            self.register_buffer(
                "selected_entropy_indices", torch.tensor(selected_indices, dtype=torch.long)
            )
            adapter_expert_count = 3 if fusion_mode == "tri_bounded_gate" else 2
            self.entropy_adapter = nn.Sequential(
                nn.LayerNorm(len(selected_indices)),
                nn.Linear(len(selected_indices), 32),
                nn.GELU(),
                nn.Linear(32, pred_len * adapter_expert_count),
            )
            nn.init.zeros_(self.entropy_adapter[-1].weight)
            nn.init.zeros_(self.entropy_adapter[-1].bias)
        self.future = nn.Sequential(nn.Linear(calendar_dim, route_dim), nn.GELU())
        self.horizon = nn.Parameter(torch.randn(pred_len, route_dim) * 0.02)
        self.trend = TrendExpert(seq_len, pred_len, variant=trend_variant)
        self.periodic = PeriodicExpert(seq_len, pred_len, variant=periodic_variant)
        self.ramp = RampExpert(enc_in, pred_len, variant=ramp_variant)
        if prior_weights is None:
            static_logits = torch.zeros(3)
        else:
            if len(prior_weights) != 3 or any(weight <= 0 for weight in prior_weights):
                raise ValueError("prior_weights must contain three positive values")
            prior = torch.as_tensor(prior_weights, dtype=torch.float32)
            prior = prior / prior.sum()
            static_logits = torch.log(prior.clamp_min(EPS))
        self.static_logits = nn.Parameter(static_logits)
        gate_input_dim = route_dim * (
            (3 if use_entropy_state else 2) if gate_mode == "horizon" else (2 if use_entropy_state else 1)
        )
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, route_dim),
            nn.GELU(),
            nn.Linear(route_dim, 3),
        )
        concat_input_dim = route_dim * (3 if use_entropy_state else 2) + 3
        self.concat_head = nn.Sequential(
            nn.Linear(concat_input_dim, route_dim),
            nn.GELU(),
            nn.Linear(route_dim, 1),
        )
        if gate_init == "zero":
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        elif gate_init != "random":
            raise ValueError("gate_init must be 'random' or 'zero'")

    def forward(self, x: Tensor, future_calendar: Tensor, state: Tensor) -> Dict[str, Tensor]:
        ramp_output = self.ramp(x)
        expert_predictions = torch.stack(
            [self.trend(x), self.periodic(x), ramp_output], dim=-1
        )
        if self.fusion_mode == "static":
            static_weights = torch.softmax(self.static_logits / self.temperature, dim=-1)
            weights = static_weights.view(1, 1, 3).expand_as(expert_predictions)
            prediction = (weights * expert_predictions).sum(dim=-1)
            return {"prediction": prediction, "weights": weights, "experts": expert_predictions}

        hist = self.history(x)
        future = self.future(future_calendar)
        state_vec = self.state(state) if self.use_entropy_state else None
        if self.fusion_mode == "concat":
            concat_parts = [
                hist[:, None, :].expand(-1, self.pred_len, -1),
            ]
            if state_vec is not None:
                concat_parts.append(state_vec[:, None, :].expand(-1, self.pred_len, -1))
            concat_parts.extend([
                future + self.horizon[None, :, :],
                expert_predictions,
            ])
            prediction = self.concat_head(torch.cat(concat_parts, dim=-1)).squeeze(-1)
            return {"prediction": prediction, "weights": None, "experts": expert_predictions}

        if self.gate_mode == "horizon":
            route_parts = [hist[:, None, :].expand(-1, self.pred_len, -1)]
            if state_vec is not None:
                route_parts.append(state_vec[:, None, :].expand(-1, self.pred_len, -1))
            route_parts.append(future + self.horizon[None, :, :])
            route = torch.cat(route_parts, dim=-1)
            logits = self.gate(route)
            if self.gate_block_size > 0:
                block = self.gate_block_size
                pooled = []
                for start in range(0, self.pred_len, block):
                    pooled.append(logits[:, start:start + block].mean(dim=1, keepdim=True))
                logits = torch.cat(pooled, dim=1)
                logits = torch.repeat_interleave(
                    logits,
                    torch.tensor(
                        [min(block, self.pred_len - start) for start in range(0, self.pred_len, block)],
                        device=logits.device,
                    ),
                    dim=1,
                )
        else:
            route_parts = [hist]
            if state_vec is not None:
                route_parts.append(state_vec)
            route = torch.cat(route_parts, dim=-1)
            logits = self.gate(route)[:, None, :].expand(-1, self.pred_len, -1)
        if self.fusion_mode == "prior_gate":
            logits = self.static_logits.view(1, 1, 3) + self.gate_residual_scale * logits
            weights = torch.softmax(logits / self.temperature, dim=-1)
        elif self.fusion_mode == "bounded_gate":
            prior = torch.softmax(self.static_logits, dim=-1).view(1, 1, 3)
            prior_pair = prior[..., :2] / prior[..., :2].sum(dim=-1, keepdim=True)
            pair_logits = logits[..., :2]
            if self.entropy_adapter is not None:
                selected_entropy = state.index_select(1, self.selected_entropy_indices)
                entropy_delta = self.entropy_adapter(selected_entropy).view(-1, self.pred_len, 2)
                pair_logits = pair_logits + self.entropy_adapter_scale * entropy_delta
            dynamic_pair = torch.softmax(
                torch.log(prior_pair.clamp_min(EPS)) + pair_logits / self.temperature,
                dim=-1,
            )
            dynamic_weights = torch.cat(
                [dynamic_pair * (1.0 - prior[..., 2:3]), prior[..., 2:3].expand_as(logits[..., 2:3])],
                dim=-1,
            )
            weights = (1.0 - self.dynamic_blend) * prior + self.dynamic_blend * dynamic_weights
        elif self.fusion_mode == "tri_bounded_gate":
            prior = torch.softmax(self.static_logits, dim=-1).view(1, 1, 3)
            if self.entropy_adapter is not None:
                selected_entropy = state.index_select(1, self.selected_entropy_indices)
                entropy_delta = self.entropy_adapter(selected_entropy).view(
                    -1, self.pred_len, 3
                )
                logits = logits + self.entropy_adapter_scale * entropy_delta
            dynamic_weights = torch.softmax(
                torch.log(prior.clamp_min(EPS)) + logits / self.temperature,
                dim=-1,
            )
            weights = (1.0 - self.dynamic_blend) * prior + self.dynamic_blend * dynamic_weights
        elif self.fusion_mode == "capped_prior_gate":
            prior = torch.softmax(self.static_logits, dim=-1).view(1, 1, 3)
            prior_pair = prior[..., :2] / prior[..., :2].sum(dim=-1, keepdim=True)
            pair_logits = torch.log(prior_pair.clamp_min(EPS)) + self.gate_residual_scale * logits[..., :2]
            dynamic_pair = torch.softmax(pair_logits / self.temperature, dim=-1)
            prior_ramp = prior[..., 2].clamp(EPS, 1.0 - EPS)
            prior_ramp_logit = torch.logit(prior_ramp)
            ramp_weight = torch.sigmoid(
                prior_ramp_logit + self.gate_residual_scale * logits[..., 2]
            ).clamp(max=self.ramp_cap).unsqueeze(-1)
            tp_mass = 1.0 - ramp_weight
            weights = torch.cat([dynamic_pair * tp_mass, ramp_weight], dim=-1)
        elif self.fusion_mode == "ramp_residual":
            prior = torch.softmax(self.static_logits, dim=-1).view(1, 1, 3)
            prior_pair = prior[..., :2] / prior[..., :2].sum(dim=-1, keepdim=True)
            pair_logits = torch.log(prior_pair.clamp_min(EPS)) + self.gate_residual_scale * logits[..., :2]
            dynamic_pair = torch.softmax(pair_logits / self.temperature, dim=-1)
            tp_mass = 1.0 - self.ramp_residual_alpha
            weights = torch.cat(
                [dynamic_pair * tp_mass, torch.full_like(logits[..., 2:3], self.ramp_residual_alpha)],
                dim=-1,
            )
            # The ramp expert predicts a delta around the last observed target;
            # it is an additive correction, not a third full forecast component.
            prediction = (
                (dynamic_pair * expert_predictions[..., :2]).sum(dim=-1)
                + self.ramp_residual_alpha * expert_predictions[..., 2]
            )
            return {
                "prediction": prediction,
                "weights": weights,
                "experts": expert_predictions,
                "ramp_delta": expert_predictions[..., 2],
            }
        else:
            weights = torch.softmax(logits / self.temperature, dim=-1)
        prediction = (weights * expert_predictions).sum(dim=-1)
        return {"prediction": prediction, "weights": weights, "experts": expert_predictions}


def forecast_loss(
    prediction: Tensor,
    target: Tensor,
    ramp_quantile: float,
) -> Tensor:
    """Forecast objective shared by individual experts and the mixture."""
    base = F.smooth_l1_loss(prediction, target)
    if prediction.shape[1] > 1:
        true_diff = target[:, 1:] - target[:, :-1]
        pred_diff = prediction[:, 1:] - prediction[:, :-1]
        slope = torch.abs(true_diff - pred_diff).mean()
        ramp_mask = (torch.abs(true_diff) >= ramp_quantile).float()
        ramp = (F.smooth_l1_loss(pred_diff, true_diff, reduction="none") * (1.0 + ramp_mask)).mean()
    else:
        slope = prediction.new_zeros(())
        ramp = prediction.new_zeros(())
    return base + 0.2 * slope + 0.3 * ramp


def horizon_loss_weights(
    horizon: int,
    mode: str = "uniform",
    power: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Return normalized horizon weights for WTR-aware objectives."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if mode not in {"uniform", "early", "late"}:
        raise ValueError("horizon weight mode must be uniform, early or late")
    if power <= 0:
        raise ValueError("horizon weight power must be positive")
    if mode == "uniform":
        values = torch.ones(horizon, device=device)
    else:
        positions = torch.linspace(0.0, 1.0, horizon, device=device)
        if mode == "early":
            values = (1.0 - positions + 1.0 / horizon).pow(power)
        else:
            values = (positions + 1.0 / horizon).pow(power)
    return values / values.mean().clamp_min(EPS)


def combined_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    ramp_quantile: float,
    balance_weight: float = 0.01,
    horizon_balance_weight: float = 0.001,
    wtr_loss_weight: float = 0.0,
    wtr_target_range: Optional[float] = None,
    wtr_temperature: float = 0.01,
    expert_predictions: Optional[Tensor] = None,
    route_loss_weight: float = 0.0,
    route_temperature: float = 0.1,
    route_expert_count: int = 3,
    wtr_horizon_mode: str = "uniform",
    wtr_horizon_power: float = 1.0,
    route_target_mode: str = "mae",
) -> Tensor:
    base = forecast_loss(prediction, target, ramp_quantile)
    usage = weights.mean(dim=(0, 1))
    balance = ((usage - 1.0 / 3.0) ** 2).sum()
    horizon_usage = weights.mean(dim=0)
    horizon_balance = ((horizon_usage - 1.0 / 3.0) ** 2).sum(dim=-1).mean()
    wtr_surrogate = prediction.new_zeros(())
    if wtr_loss_weight > 0:
        if wtr_target_range is None or wtr_target_range <= EPS:
            raise ValueError("A positive training target range is required for WTR loss")
        relative_error = (prediction - target).abs() / wtr_target_range
        temperature = max(wtr_temperature, EPS)
        success = [
            torch.sigmoid((threshold - relative_error) / temperature)
            for threshold in (0.05, 0.10, 0.15)
        ]
        horizon_weights = horizon_loss_weights(
            prediction.shape[1],
            mode=wtr_horizon_mode,
            power=wtr_horizon_power,
            device=prediction.device,
        ).view(1, -1)
        success = [(value * horizon_weights).mean() for value in success]
        weighted_success = 0.5 * success[0] + 0.3 * success[1] + 0.2 * success[2]
        wtr_surrogate = 1.0 - weighted_success
    route_loss = prediction.new_zeros(())
    if route_loss_weight > 0:
        if expert_predictions is None:
            raise ValueError("expert_predictions are required for route supervision")
        routed_experts = expert_predictions[..., :route_expert_count]
        routed_weights = weights[..., :route_expert_count]
        routed_weights = routed_weights / routed_weights.sum(dim=-1, keepdim=True).clamp_min(EPS)
        if route_target_mode == "mae":
            expert_error = (routed_experts - target.unsqueeze(-1)).abs()
            route_target = torch.softmax(
                -expert_error / max(route_temperature, EPS), dim=-1
            ).detach()
        elif route_target_mode == "wtr":
            relative_expert_error = (
                (routed_experts - target.unsqueeze(-1)).abs()
                / max(wtr_target_range or 1.0, EPS)
            )
            expert_success = (
                0.5 * torch.sigmoid((0.05 - relative_expert_error) / max(route_temperature, EPS))
                + 0.3 * torch.sigmoid((0.10 - relative_expert_error) / max(route_temperature, EPS))
                + 0.2 * torch.sigmoid((0.15 - relative_expert_error) / max(route_temperature, EPS))
            )
            route_target = torch.softmax(
                expert_success / max(route_temperature, EPS), dim=-1
            ).detach()
        else:
            raise ValueError("route_target_mode must be mae or wtr")
        route_loss = (
            route_target
            * (torch.log(route_target.clamp_min(EPS)) - torch.log(routed_weights.clamp_min(EPS)))
        ).sum(dim=-1).mean()
    return (
        base
        + balance_weight * balance
        + horizon_balance_weight * horizon_balance
        + wtr_loss_weight * wtr_surrogate
        + route_loss_weight * route_loss
    )


def independent_expert_loss(
    expert_predictions: Tensor,
    target: Tensor,
    ramp_quantile: float,
    ramp_residual: bool = False,
    last_target: Optional[Tensor] = None,
) -> Tensor:
    """Supervise each expert directly to remove coalition compensation."""
    if expert_predictions.ndim != 3:
        raise ValueError("expert_predictions must have shape [batch, horizon, expert]")
    losses = [
        forecast_loss(expert_predictions[..., expert], target, ramp_quantile)
        for expert in range(2)
    ]
    if ramp_residual:
        if last_target is None:
            raise ValueError("last_target is required for residual ramp supervision")
        ramp_target = target - last_target
        losses.append(
            forecast_loss(expert_predictions[..., 2], ramp_target, ramp_quantile)
        )
    else:
        losses.append(forecast_loss(expert_predictions[..., 2], target, ramp_quantile))
    return torch.stack(losses).mean()


def train_one_epoch(
    model: CESHMoE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    ramp_quantile: float,
    balance_weight: float,
    horizon_balance_weight: float,
    fusion: str = "gate",
    wtr_loss_weight: float = 0.0,
    wtr_target_range: Optional[float] = None,
    wtr_temperature: float = 0.01,
    route_loss_weight: float = 0.0,
    route_temperature: float = 0.1,
    route_expert_count: int = 3,
    wtr_horizon_mode: str = "uniform",
    wtr_horizon_power: float = 1.0,
    route_target_mode: str = "mae",
) -> float:
    model.train()
    total = 0.0
    count = 0
    for batch in loader:
        non_blocking = device.startswith("cuda")
        x = batch["x"].to(device, non_blocking=non_blocking).float()
        future = batch["future_calendar"].to(device, non_blocking=non_blocking).float()
        y = batch["y"].to(device, non_blocking=non_blocking).float()
        state = batch["state"].to(device, non_blocking=non_blocking).float()
        output = model(x, future, state)
        if fusion == "uniform":
            uniform_weights = torch.full_like(output["weights"], 1.0 / 3.0)
            prediction = (uniform_weights * output["experts"]).sum(dim=-1)
            weights = uniform_weights
            loss = combined_loss(
                prediction,
                y,
                weights,
                ramp_quantile=ramp_quantile,
                balance_weight=balance_weight,
                horizon_balance_weight=horizon_balance_weight,
                wtr_loss_weight=wtr_loss_weight,
                wtr_target_range=wtr_target_range,
                wtr_temperature=wtr_temperature,
                expert_predictions=output["experts"],
                route_loss_weight=route_loss_weight,
                route_temperature=route_temperature,
                route_expert_count=route_expert_count,
                wtr_horizon_mode=wtr_horizon_mode,
                wtr_horizon_power=wtr_horizon_power,
                route_target_mode=route_target_mode,
            )
        elif fusion == "independent":
            last_target = x[:, -1, -1].unsqueeze(1)
            loss = independent_expert_loss(
                output["experts"],
                y,
                ramp_quantile=ramp_quantile,
                ramp_residual=model.ramp_residual,
                last_target=last_target,
            )
        elif fusion == "concat":
            loss = forecast_loss(output["prediction"], y, ramp_quantile=ramp_quantile)
        else:
            prediction = output["prediction"]
            weights = output["weights"]
            loss = combined_loss(
                prediction,
                y,
                weights,
                ramp_quantile=ramp_quantile,
                balance_weight=balance_weight,
                horizon_balance_weight=horizon_balance_weight,
                wtr_loss_weight=wtr_loss_weight,
                wtr_target_range=wtr_target_range,
                wtr_temperature=wtr_temperature,
                expert_predictions=output["experts"],
                route_loss_weight=route_loss_weight,
                route_temperature=route_temperature,
                route_expert_count=route_expert_count,
                wtr_horizon_mode=wtr_horizon_mode,
                wtr_horizon_power=wtr_horizon_power,
                route_target_mode=route_target_mode,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total += float(loss.detach()) * len(x)
        count += len(x)
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: CESHMoE,
    loader: DataLoader,
    device: str,
    scaler: Optional[NumpyScaler] = None,
    target_idx: Optional[int] = None,
    include_horizon_weights: bool = False,
) -> Dict[str, object]:
    model.eval()
    errors = []
    absolute = []
    targets = []
    expert_weights = []
    non_blocking = device.startswith("cuda")
    for batch in loader:
        output = model(
            batch["x"].to(device, non_blocking=non_blocking).float(),
            batch["future_calendar"].to(device, non_blocking=non_blocking).float(),
            batch["state"].to(device, non_blocking=non_blocking).float(),
        )
        if output["weights"] is not None:
            expert_weights.append(output["weights"].cpu())
        error = output["prediction"] - batch["y"].to(device, non_blocking=non_blocking).float()
        errors.append(error.cpu())
        absolute.append(error.abs().cpu())
        targets.append(batch["y"].detach().cpu().float())
    error = torch.cat(errors)
    absolute = torch.cat(absolute)
    target_values = torch.cat(targets)
    metrics = {
        "mse": float((error**2).mean()),
        "rmse": float(torch.sqrt((error**2).mean())),
        "mae": float(absolute.mean()),
    }
    if expert_weights:
        weights = torch.cat(expert_weights)
        dominant = weights.argmax(dim=-1)
        metrics.update(
            {
                "trend_weight": float(weights[..., 0].mean()),
                "periodic_weight": float(weights[..., 1].mean()),
                "ramp_weight": float(weights[..., 2].mean()),
                "trend_weight_std": float(weights[..., 0].std()),
                "periodic_weight_std": float(weights[..., 1].std()),
                "ramp_weight_std": float(weights[..., 2].std()),
                "trend_sample_std": float(weights[..., 0].mean(dim=1).std()),
                "periodic_sample_std": float(weights[..., 1].mean(dim=1).std()),
                "ramp_sample_std": float(weights[..., 2].mean(dim=1).std()),
                "trend_horizon_std": float(weights[..., 0].mean(dim=0).std()),
                "periodic_horizon_std": float(weights[..., 1].mean(dim=0).std()),
                "ramp_horizon_std": float(weights[..., 2].mean(dim=0).std()),
                "trend_dominant": float((dominant == 0).float().mean()),
                "periodic_dominant": float((dominant == 1).float().mean()),
                "ramp_dominant": float((dominant == 2).float().mean()),
            }
        )
        if include_horizon_weights:
            metrics["horizon_weights"] = weights.mean(dim=0).tolist()
    if scaler is not None:
        if target_idx is None:
            raise ValueError("target_idx is required when reporting original-scale metrics")
        raw_error = error * float(scaler.std[target_idx])
        raw_target = target_values * float(scaler.std[target_idx]) + float(scaler.mean[target_idx])
        target_range = float(raw_target.max() - raw_target.min())
        if target_range <= EPS:
            raise ValueError("Cannot compute original-scale metrics because the target range is zero")
        mape_floor = max(0.01 * target_range, 1e-6)
        metrics.update(
            {
                "mse_raw": float((raw_error**2).mean()),
                "rmse_raw": float(torch.sqrt((raw_error**2).mean())),
                "mae_raw": float(raw_error.abs().mean()),
                "mape": float(
                    (raw_error.abs() / torch.clamp(raw_target.abs(), min=mape_floor)).mean() * 100.0
                ),
                "r2": float(
                    1.0
                    - (raw_error**2).sum()
                    / ((raw_target - raw_target.mean()) ** 2).sum().clamp_min(EPS)
                ),
            }
        )
        relative_error = raw_error.abs() / target_range
        metrics.update(
            {
                # Report WTR in percentage points, matching the paper's WTR (%).
                "wtr5": float((relative_error <= 0.05).float().mean() * 100.0),
                "wtr10": float((relative_error <= 0.10).float().mean() * 100.0),
                "wtr15": float((relative_error <= 0.15).float().mean() * 100.0),
            }
        )
        metrics["weighted_wtr"] = (
            0.5 * metrics["wtr5"]
            + 0.3 * metrics["wtr10"]
            + 0.2 * metrics["wtr15"]
        )
    return metrics


def smoke_test() -> None:
    """Check all tensor contracts without requiring a downloaded ETT CSV."""
    batch, seq_len, pred_len, channels, entropy_dim = 4, 168, 24, 7, state_dim(7, 3)
    x = torch.randn(batch, seq_len, channels)
    future = torch.randn(batch, pred_len, 6)
    state = torch.randn(batch, entropy_dim)
    target = torch.randn(batch, pred_len)
    prior_weights = (0.853, 0.142, 0.004)
    outputs = {}
    for fusion_mode, selective_entropy in (
        ("gate", False),
        ("bounded_gate", False),
        ("tri_bounded_gate", True),
        ("capped_prior_gate", False),
        ("ramp_residual", False),
    ):
        model = CESHMoE(
            seq_len,
            pred_len,
            channels,
            entropy_dim,
            fusion_mode=fusion_mode,
            prior_weights=prior_weights if fusion_mode != "gate" else None,
            dynamic_blend=0.5,
            ramp_cap=0.02,
            ramp_residual_alpha=0.02,
            selective_entropy=selective_entropy,
        )
        output = model(x, future, state)
        assert output["prediction"].shape == (batch, pred_len)
        assert output["weights"].shape == (batch, pred_len, 3)
        assert output["experts"].shape == (batch, pred_len, 3)
        assert torch.allclose(
            output["weights"].sum(-1), torch.ones(batch, pred_len), atol=1e-5
        )
        loss = combined_loss(
            output["prediction"],
            target,
            output["weights"],
            ramp_quantile=1.0,
            expert_predictions=output["experts"],
            route_loss_weight=0.005,
            route_temperature=0.1,
            route_expert_count=3,
        )
        loss.backward()
        assert model.trend.linear.weight.grad is not None
        assert model.gate[-1].weight.grad is not None
        outputs[fusion_mode] = output

    bounded_prior_ramp = prior_weights[2] / sum(prior_weights)
    assert torch.allclose(
        outputs["bounded_gate"]["weights"][..., 2],
        torch.full((batch, pred_len), bounded_prior_ramp),
        atol=1e-6,
    )
    assert float(outputs["tri_bounded_gate"]["weights"][..., 2].std()) > 0.0
    assert float(outputs["capped_prior_gate"]["weights"][..., 2].max()) <= 0.02 + 1e-6
    assert torch.allclose(
        outputs["ramp_residual"]["weights"][..., 2],
        torch.full((batch, pred_len), 0.02),
        atol=1e-6,
    )
    residual_model = CESHMoE(
        seq_len,
        pred_len,
        channels,
        entropy_dim,
        fusion_mode="ramp_residual",
        prior_weights=prior_weights,
        gate_residual_scale=0.05,
        ramp_residual_alpha=0.02,
    )
    residual_output = residual_model(x, future, state)
    tp_prediction = (
        residual_output["weights"][..., :2]
        * residual_output["experts"][..., :2]
    ).sum(dim=-1) / (1.0 - 0.02)
    expected_prediction = tp_prediction + 0.02 * residual_output["experts"][..., 2]
    assert torch.allclose(residual_output["prediction"], expected_prediction, atol=1e-5)
    print("smoke test passed", {name: tuple(output["weights"].shape) for name, output in outputs.items()})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Keep seed-controlled initialization/order while allowing faster CUDA
    # kernels; tiny floating-point differences between runs are acceptable.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_csv(data_dir: str, dataset: str) -> Path:
    """Resolve common ETT layouts without requiring a shell-side special case."""
    root = Path(data_dir)
    candidates = [
        root / f"{dataset}.csv",
        root / dataset / f"{dataset}.csv",
        root / "ETT-small" / f"{dataset}.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {dataset}.csv. Tried: {tried}")


def rounded_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """Round only displayed metrics; training keeps full precision."""
    return {key: round(value, 3) for key, value in metrics.items()}


def select_validation_candidate(
    candidates: Sequence[Dict[str, object]],
    selection_metric: str,
    mse_tolerance: float,
) -> Dict[str, object]:
    if not candidates:
        raise ValueError("No validation candidates are available for model selection")
    if selection_metric == "mse":
        return min(candidates, key=lambda item: (float(item["mse"]), int(item["epoch"])))
    if selection_metric == "weighted_wtr":
        return max(
            candidates,
            key=lambda item: (float(item["weighted_wtr"]), -float(item["mse"]), -int(item["epoch"])),
        )
    minimum_mse = min(float(item["mse"]) for item in candidates)
    eligible = [
        item for item in candidates
        if float(item["mse"]) <= minimum_mse * (1.0 + mse_tolerance)
    ]
    return max(
        eligible,
        key=lambda item: (float(item["weighted_wtr"]), -float(item["mse"]), -int(item["epoch"])),
    )


def configure_stage_trainability(
    model: CESHMoE,
    stage_name: str,
    stage2_scope: str,
    stage3_scope: str,
) -> None:
    """Set the exact parameter scope for each staged-training phase."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage_name == "expert_pretrain":
        trainable_modules = (model.trend, model.periodic, model.ramp)
    elif stage_name == "gate_frozen_experts":
        if stage2_scope == "gate_only":
            trainable_modules = (model.gate,)
            if model.entropy_adapter is not None:
                trainable_modules = trainable_modules + (model.entropy_adapter,)
        elif stage2_scope == "fusion_only":
            trainable_modules = () if model.fusion_mode == "static" else (model.concat_head,)
            if model.fusion_mode == "static":
                model.static_logits.requires_grad = True
        else:
            trainable_modules = (model.history, model.state, model.future, model.gate)
            model.horizon.requires_grad = True
            if model.entropy_adapter is not None:
                trainable_modules = trainable_modules + (model.entropy_adapter,)
    elif stage_name == "joint_finetune":
        if stage3_scope == "joint":
            for parameter in model.parameters():
                parameter.requires_grad = True
            return
        if stage3_scope == "gate_only":
            trainable_modules = (model.gate,)
            if model.entropy_adapter is not None:
                trainable_modules = trainable_modules + (model.entropy_adapter,)
        elif stage3_scope == "fusion_only":
            trainable_modules = () if model.fusion_mode == "static" else (model.concat_head,)
            if model.fusion_mode == "static":
                model.static_logits.requires_grad = True
        else:
            raise ValueError(f"Unknown stage-3 scope: {stage3_scope}")
    else:
        raise ValueError(f"Unknown training stage: {stage_name}")

    for module in trainable_modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--loader_seed", type=int, default=None,
        help="Fixed train DataLoader shuffle seed; omit to use the global RNG.",
    )
    parser.add_argument(
        "--master_seed", type=int, default=None,
        help="Use one seed for initialization, training and DataLoader order. "
        "Overrides --seed, --init_seed and --loader_seed.",
    )
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument(
        "--stage1_lr", type=float, default=None,
        help="Expert pretraining learning rate; defaults to --lr.",
    )
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--init_seed", type=int, default=None,
        help="Optional seed used only for model initialization; training is reset to --seed afterward.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--k", type=int, default=2, help="Reserved for the optional sparse router.")
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--horizon_balance_weight", type=float, default=0.001)
    parser.add_argument(
        "--wtr_loss_weight", type=float, default=0.0,
        help="Weight of a smooth training-only Weighted WTR surrogate (default: disabled).",
    )
    parser.add_argument(
        "--wtr_temperature", type=float, default=0.01,
        help="Smooth WTR threshold transition width as a fraction of the training target range.",
    )
    parser.add_argument(
        "--route_loss_weight", type=float, default=0.0,
        help="Weight of per-horizon soft expert-oracle routing supervision.",
    )
    parser.add_argument(
        "--route_temperature", type=float, default=0.1,
        help="Temperature used to convert expert errors into routing targets.",
    )
    parser.add_argument(
        "--route_expert_count", type=int, choices=[2, 3], default=3,
        help="Number of leading experts included in routing supervision.",
    )
    parser.add_argument(
        "--route_target_mode", choices=["mae", "wtr"], default="mae",
        help="Soft route target based on expert MAE or smooth WTR utility.",
    )
    parser.add_argument(
        "--selection_metric",
        choices=["mse", "weighted_wtr", "constrained_wtr"],
        default="mse",
        help="Validation-only checkpoint selection rule.",
    )
    parser.add_argument(
        "--selection_mse_tolerance", type=float, default=0.01,
        help="Allowed validation-MSE increase for constrained_wtr selection (default: 0.01).",
    )
    parser.add_argument("--gate_init", choices=["random", "zero"], default="random")
    parser.add_argument(
        "--gate_mode", choices=["horizon", "global"], default="horizon",
        help="Horizon-wise gate or one global expert mixture shared across all horizons.",
    )
    parser.add_argument(
        "--gate_block_size", type=int, default=0,
        help="Average horizon logits within fixed blocks; 0 keeps the original gate.",
    )
    parser.add_argument(
        "--wtr_horizon_mode", choices=["uniform", "early", "late"], default="uniform",
        help="Horizon weighting used by the smooth WTR training surrogate.",
    )
    parser.add_argument(
        "--wtr_horizon_power", type=float, default=1.0,
        help="Power applied to early/late horizon weights.",
    )
    parser.add_argument(
        "--trend_variant", choices=["v1", "v2"], default="v1",
        help="Trend expert variant.",
    )
    parser.add_argument(
        "--periodic_variant", choices=["v1", "v2", "v3"], default="v1",
        help="Periodic expert variant; v2/v3 add seasonal anchors.",
    )
    parser.add_argument(
        "--ramp_variant", choices=["v1", "v2", "v3"], default="v1",
        help="Ramp expert variant; v2 is residual and v3 adds a wider TCN.",
    )
    parser.add_argument(
        "--disable_entropy_state", action="store_true",
        help="Do not pass the entropy-state representation into the gate.",
    )
    parser.add_argument(
        "--stage1_fusion", choices=["uniform", "independent"], default="uniform",
        help="Stage-1 objective: uniform ensemble or independent expert supervision.",
    )
    parser.add_argument(
        "--save_stage1_checkpoint", default=None,
        help="Save the complete model state after stage 1 for shared-expert runs.",
    )
    parser.add_argument(
        "--load_stage1_checkpoint", default=None,
        help="Load a shared stage-1 model state and skip expert pretraining.",
    )
    parser.add_argument(
        "--load_expert_checkpoint", default=None,
        help="Load only trend/periodic/ramp weights from a stage-1 checkpoint and skip expert pretraining.",
    )
    parser.add_argument(
        "--save_best_checkpoint", default=None,
        help="Save the validation-selected final model state.",
    )
    parser.add_argument(
        "--stage1_only", action="store_true",
        help="Run only stage 1 and save its checkpoint when requested.",
    )
    parser.add_argument(
        "--stage2_scope", choices=["routing", "gate_only", "fusion_only"], default="routing",
        help="Stage-2 trainable scope: routing encoders plus gate, or gate only.",
    )
    parser.add_argument(
        "--stage3_scope", choices=["joint", "gate_only", "fusion_only"], default="joint",
        help="Stage-3 trainable scope: all parameters, or gate only.",
    )
    parser.add_argument(
        "--fusion_mode",
        choices=[
            "gate", "concat", "static", "prior_gate", "bounded_gate", "tri_bounded_gate",
            "capped_prior_gate", "ramp_residual",
        ],
        default="gate",
        help="Dynamic routing, entropy concatenation, global weights, or prior-guided dynamic routing.",
    )
    parser.add_argument(
        "--prior_weights", default=None,
        help="Comma-separated trend,periodic,ramp prior weights for prior_gate.",
    )
    parser.add_argument(
        "--gate_residual_scale", type=float, default=0.1,
        help="Dynamic-logit scale around the static prior in prior_gate mode.",
    )
    parser.add_argument(
        "--dynamic_blend", type=float, default=0.1,
        help="Maximum dynamic mixture contribution in bounded_gate modes.",
    )
    parser.add_argument(
        "--ramp_cap", type=float, default=0.02,
        help="Maximum ramp weight for capped_prior_gate mode.",
    )
    parser.add_argument(
        "--ramp_residual_alpha", type=float, default=0.02,
        help="Fixed residual contribution for ramp_residual mode.",
    )
    parser.add_argument(
        "--selective_entropy", action="store_true",
        help="Inject only target PE/SpE/SampEn/ApEn/slope/ramp features through a weak adapter.",
    )
    parser.add_argument(
        "--entropy_adapter_scale", type=float, default=0.01,
        help="Scale of the selective entropy logit correction.",
    )
    parser.add_argument("--stage2_lr", type=float, default=1e-5)
    parser.add_argument(
        "--stage1_epochs", type=int, default=30,
        help="Expert pretraining epochs (default: 30).",
    )
    parser.add_argument(
        "--stage2_epochs", type=int, default=30,
        help="Maximum frozen-expert gate training epochs (default: 30).",
    )
    parser.add_argument(
        "--stage2_patience", type=int, default=5,
        help="Stage-2 gate early-stopping patience (default: 5).",
    )
    parser.add_argument(
        "--stage3_patience", type=int, default=10,
        help="Stage-3 joint fine-tuning patience (default: 10).",
    )
    parser.add_argument(
        "--finetune_lr", type=float, default=None,
        help="Stage-3 joint fine-tuning learning rate; defaults to lr/10.",
    )
    parser.add_argument(
        "--suppress_horizon_weights", action="store_true",
        help="Do not print the full per-horizon mean gate-weight array.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.master_seed is not None:
        args.seed = args.master_seed
        args.init_seed = args.master_seed
        args.loader_seed = args.master_seed
    if min(
        args.epochs, args.batch_size, args.patience, args.min_epochs,
        args.stage1_epochs, args.stage2_epochs,
        args.stage2_patience, args.stage3_patience,
    ) < 1:
        parser.error("training, stage, patience and batch arguments must be positive")
    if args.horizon_balance_weight < 0:
        parser.error("--horizon_balance_weight must be non-negative")
    if args.wtr_loss_weight < 0:
        parser.error("--wtr_loss_weight must be non-negative")
    if args.wtr_temperature <= 0:
        parser.error("--wtr_temperature must be positive")
    if args.route_loss_weight < 0:
        parser.error("--route_loss_weight must be non-negative")
    if args.route_temperature <= 0:
        parser.error("--route_temperature must be positive")
    if args.gate_block_size < 0:
        parser.error("--gate_block_size must be non-negative")
    if args.wtr_horizon_power <= 0:
        parser.error("--wtr_horizon_power must be positive")
    if args.gate_residual_scale <= 0:
        parser.error("--gate_residual_scale must be positive")
    if not 0 < args.dynamic_blend <= 1:
        parser.error("--dynamic_blend must be in (0, 1]")
    if args.entropy_adapter_scale <= 0:
        parser.error("--entropy_adapter_scale must be positive")
    if not 0 < args.ramp_cap <= 1:
        parser.error("--ramp_cap must be in (0, 1]")
    if not 0 <= args.ramp_residual_alpha <= 1:
        parser.error("--ramp_residual_alpha must be in [0, 1]")
    if args.selective_entropy and args.fusion_mode not in {"bounded_gate", "tri_bounded_gate"}:
        parser.error(
            "--selective_entropy currently requires --fusion_mode bounded_gate or tri_bounded_gate"
        )
    if args.selection_mse_tolerance < 0:
        parser.error("--selection_mse_tolerance must be non-negative")
    if args.stage2_lr <= 0:
        parser.error("--stage2_lr must be positive")
    if args.stage1_lr is not None and args.stage1_lr <= 0:
        parser.error("--stage1_lr must be positive")
    if args.min_epochs > args.epochs:
        parser.error("--min_epochs cannot be greater than --epochs")
    stage1_epochs = args.stage1_epochs
    stage2_epochs = args.stage2_epochs
    stage3_epochs = args.epochs - stage1_epochs - stage2_epochs
    if stage1_epochs < 1 or stage2_epochs < 1 or stage3_epochs < 1:
        parser.error("stage1_epochs + stage2_epochs must be less than epochs")
    if args.finetune_lr is not None and args.finetune_lr <= 0:
        parser.error("--finetune_lr must be positive")
    if args.stage1_only and (args.load_stage1_checkpoint or args.load_expert_checkpoint):
        parser.error("--stage1_only cannot be combined with a stage-1 checkpoint")
    if args.load_stage1_checkpoint and args.load_expert_checkpoint:
        parser.error("Use only one stage-1 checkpoint loading option")
    if args.stage1_only and not args.save_stage1_checkpoint:
        parser.error("--stage1_only requires --save_stage1_checkpoint")
    prior_weights = None
    if args.prior_weights is not None:
        try:
            prior_weights = [float(value.strip()) for value in args.prior_weights.split(",")]
        except ValueError:
            parser.error("--prior_weights must contain three comma-separated numbers")
        if len(prior_weights) != 3 or any(value <= 0 for value in prior_weights):
            parser.error("--prior_weights must contain three positive values")
    if args.fusion_mode in {
        "prior_gate", "bounded_gate", "tri_bounded_gate", "capped_prior_gate", "ramp_residual"
    } and prior_weights is None:
        parser.error("prior-guided fusion modes require --prior_weights")
    set_seed(args.init_seed if args.init_seed is not None else args.seed)
    csv_path = Path(args.csv) if args.csv else resolve_csv(args.data_dir, args.dataset)
    config = ETTConfig(
        str(csv_path),
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        loader_seed=args.loader_seed,
    )
    data = ETTDataModule(config)
    loaders = data.loaders()
    num_train = data.borders[0]
    train_target_range = float(
        np.ptp(data.values[:num_train, data.target_idx])
    )
    if train_target_range <= EPS:
        parser.error("Training target range must be positive")
    model = CESHMoE(
        config.seq_len,
        config.pred_len,
        enc_in=len(data.columns),
        entropy_dim=state_dim(len(data.columns), len(config.entropy_scales)),
        gate_init=args.gate_init,
        use_entropy_state=not args.disable_entropy_state,
        gate_mode=args.gate_mode,
        fusion_mode=args.fusion_mode,
        prior_weights=prior_weights,
        gate_residual_scale=args.gate_residual_scale,
        dynamic_blend=args.dynamic_blend,
        ramp_cap=args.ramp_cap,
        ramp_residual_alpha=args.ramp_residual_alpha,
        selective_entropy=args.selective_entropy,
        entropy_adapter_scale=args.entropy_adapter_scale,
        trend_variant=args.trend_variant,
        periodic_variant=args.periodic_variant,
        ramp_variant=args.ramp_variant,
        gate_block_size=args.gate_block_size,
    )
    if args.init_seed is not None and args.init_seed != args.seed and not (
        args.load_stage1_checkpoint or args.load_expert_checkpoint
    ):
        expert_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
            if key.startswith(("trend.", "periodic.", "ramp."))
        }
        set_seed(args.seed)
        model = CESHMoE(
            config.seq_len,
            config.pred_len,
            enc_in=len(data.columns),
            entropy_dim=state_dim(len(data.columns), len(config.entropy_scales)),
            gate_init=args.gate_init,
            use_entropy_state=not args.disable_entropy_state,
            gate_mode=args.gate_mode,
            fusion_mode=args.fusion_mode,
            prior_weights=prior_weights,
            gate_residual_scale=args.gate_residual_scale,
            dynamic_blend=args.dynamic_blend,
            ramp_cap=args.ramp_cap,
            ramp_residual_alpha=args.ramp_residual_alpha,
            selective_entropy=args.selective_entropy,
            entropy_adapter_scale=args.entropy_adapter_scale,
            trend_variant=args.trend_variant,
            periodic_variant=args.periodic_variant,
            ramp_variant=args.ramp_variant,
            gate_block_size=args.gate_block_size,
        )
        model.load_state_dict(expert_state, strict=False)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    if args.init_seed is not None:
        set_seed(args.seed)
    checkpoint_arg = args.load_stage1_checkpoint or args.load_expert_checkpoint
    if checkpoint_arg:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.is_file():
            parser.error(f"Stage-1 checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict):
            parser.error("Stage-1 checkpoint must contain a model state dictionary")
        if args.load_expert_checkpoint:
            expert_state = {
                key: value for key, value in checkpoint.items()
                if key.startswith(("trend.", "periodic.", "ramp."))
            }
            if not expert_state:
                parser.error("Expert checkpoint does not contain trend/periodic/ramp weights")
            missing, _ = model.load_state_dict(expert_state, strict=False)
            required = [key for key in missing if key.startswith(("trend.", "periodic.", "ramp."))]
            if required:
                parser.error(f"Invalid expert checkpoint; missing expert keys: {required[:3]}")
            print(f"loaded_expert_checkpoint={checkpoint_path}")
        else:
            try:
                model.load_state_dict(checkpoint, strict=True)
            except (RuntimeError, TypeError) as exc:
                # Older gate-only checkpoints predate the optional concat head.
                # Accept them only when those are the sole missing parameters.
                try:
                    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
                except (RuntimeError, TypeError):
                    parser.error(f"Invalid stage-1 checkpoint: {exc}")
                allowed_missing = {
                    "static_logits",
                    "concat_head.0.weight", "concat_head.0.bias",
                    "concat_head.2.weight", "concat_head.2.bias",
                }
                if set(missing) - allowed_missing or unexpected:
                    parser.error(f"Invalid stage-1 checkpoint: {exc}")
            print(f"loaded_stage1_checkpoint={checkpoint_path}")
        model.to(device)
    stage1_lr = args.stage1_lr if args.stage1_lr is not None else args.lr
    stage3_lr = args.finetune_lr if args.finetune_lr is not None else args.lr / 10.0
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stage2_best_state = None
    validation_candidates = []
    print(
        f"dataset={args.dataset} csv={csv_path} seq_len={config.seq_len} pred_len={config.pred_len} "
        f"device={device} seed={args.seed} loader_seed={args.loader_seed} gate_init={args.gate_init} "
        f"gate_mode={args.gate_mode} fusion_mode={args.fusion_mode} "
        f"gate_block_size={args.gate_block_size} "
        f"entropy_state={'disabled' if args.disable_entropy_state else 'enabled'} "
        f"experts={args.trend_variant}/{args.periodic_variant}/{args.ramp_variant}"
    )
    print(
        f"staged_training=expert_pretrain:{stage1_epochs} "
        f"gate_frozen_experts:{stage2_epochs} joint_finetune:{stage3_epochs} "
        f"stage1_lr={stage1_lr:.2e} "
        f"stage2_lr={args.stage2_lr:.2e} finetune_lr={stage3_lr:.2e} "
        f"stage2_scope={args.stage2_scope} "
        f"stage3_scope={args.stage3_scope} "
        f"stage1_fusion={args.stage1_fusion} "
        f"horizon_balance_weight={args.horizon_balance_weight:.3f} "
        f"wtr_loss_weight={args.wtr_loss_weight:.3f} "
        f"route_loss_weight={args.route_loss_weight:.3f} "
        f"route_target_mode={args.route_target_mode} "
        f"wtr_horizon_mode={args.wtr_horizon_mode} "
        f"wtr_horizon_power={args.wtr_horizon_power:.3f} "
        f"gate_residual_scale={args.gate_residual_scale:.3f} "
        f"dynamic_blend={args.dynamic_blend:.3f} "
        f"ramp_cap={args.ramp_cap:.3f} "
        f"ramp_residual_alpha={args.ramp_residual_alpha:.3f} "
        f"selective_entropy={'enabled' if args.selective_entropy else 'disabled'} "
        f"entropy_adapter_scale={args.entropy_adapter_scale:.3f} "
        f"selection_metric={args.selection_metric} "
        f"selection_mse_tolerance={args.selection_mse_tolerance:.3f}"
    )
    stages = []
    if not checkpoint_arg:
        stages.append(("expert_pretrain", stage1_epochs, stage1_lr, args.stage1_fusion, True))
    if not args.stage1_only:
        train_fusion = "concat" if args.fusion_mode == "concat" else "gate"
        stages.extend([
            ("gate_frozen_experts", stage2_epochs, args.stage2_lr, train_fusion, False),
            ("joint_finetune", stage3_epochs, stage3_lr, train_fusion, True),
        ])
    global_epoch = 0
    prior_initial_candidate = None
    if args.fusion_mode in {
        "prior_gate", "bounded_gate", "tri_bounded_gate", "capped_prior_gate", "ramp_residual"
    } and not args.stage1_only:
        initial_val = evaluate(model, loaders["val"], device, data.scaler, data.target_idx)
        initial_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        prior_initial_candidate = {
            "epoch": 0,
            "mse": initial_val["mse"],
            "mae": initial_val["mae"],
            "weighted_wtr": initial_val["weighted_wtr"],
            "state": initial_state,
        }
        validation_candidates.append(prior_initial_candidate)
        best_val = initial_val["mse"]
        best_epoch = 0
        print(f"prior_initial_val={rounded_metrics(initial_val)}")
    for stage_name, stage_epochs, stage_lr, fusion, train_experts in stages:
        configure_stage_trainability(model, stage_name, args.stage2_scope, args.stage3_scope)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=stage_lr,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
            threshold=1e-4, min_lr=1e-6
        )
        print(f"stage={stage_name} epochs={stage_epochs} lr={stage_lr:.2e}")
        stage_best_val = (
            float(prior_initial_candidate["mse"])
            if stage_name == "gate_frozen_experts" and prior_initial_candidate is not None
            else float("inf")
        )
        stage_stale = 0
        stage_candidates = (
            [prior_initial_candidate]
            if stage_name == "gate_frozen_experts" and prior_initial_candidate is not None
            else []
        )
        stage_wtr_loss_weight = args.wtr_loss_weight if stage_name != "expert_pretrain" else 0.0
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            loss = train_one_epoch(
                model, loaders["train"], optimizer, device,
                data.ramp_threshold, args.balance_weight,
                args.horizon_balance_weight,
                wtr_loss_weight=stage_wtr_loss_weight,
                wtr_target_range=train_target_range,
                wtr_temperature=args.wtr_temperature,
                route_loss_weight=args.route_loss_weight,
                route_temperature=args.route_temperature,
                route_expert_count=args.route_expert_count,
                wtr_horizon_mode=args.wtr_horizon_mode,
                wtr_horizon_power=args.wtr_horizon_power,
                route_target_mode=args.route_target_mode,
                fusion=fusion,
            )
            val = evaluate(model, loaders["val"], device, data.scaler, data.target_idx)
            scheduler.step(val["mse"])
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"stage={stage_name} epoch={global_epoch} lr={current_lr:.6g} "
                f"train_loss={loss:.3f} val={rounded_metrics(val)}"
            )
            # Stage 1 trains an equal-weight ensemble while the gate is still
            # untrained, so its gate-based validation score is not comparable.
            if stage_name != "expert_pretrain":
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                candidate = {
                    "epoch": global_epoch,
                    "mse": val["mse"],
                    "mae": val["mae"],
                    "weighted_wtr": val["weighted_wtr"],
                    "state": state,
                }
                stage_candidates.append(candidate)
                validation_candidates.append(candidate)
                if val["mse"] < best_val:
                    best_val = val["mse"]
                    best_epoch = global_epoch
                if val["mse"] < stage_best_val:
                    stage_best_val = val["mse"]
                    stage_stale = 0
                else:
                    stage_stale += 1
            else:
                stage_stale += 1
            patience = args.stage2_patience if stage_name == "gate_frozen_experts" else args.stage3_patience
            if stage_name == "gate_frozen_experts" and stage_stale >= patience:
                print(f"Stage-2 early stopping at epoch {global_epoch} (best_epoch={global_epoch - stage_stale})")
                break
            if stage_name == "joint_finetune" and global_epoch >= args.min_epochs and stage_stale >= patience:
                print(f"Stage-3 early stopping at epoch {global_epoch}")
                break
        if stage_name == "gate_frozen_experts" and stage_candidates:
            stage2_selected = select_validation_candidate(
                stage_candidates, args.selection_metric, args.selection_mse_tolerance
            )
            stage2_best_state = stage2_selected["state"]
            model.load_state_dict(stage2_best_state)
            model.to(device)
            print(
                f"Restored stage-2 selected state: epoch={stage2_selected['epoch']} "
                f"val_mse={float(stage2_selected['mse']):.6f} "
                f"val_mae={float(stage2_selected['mae']):.6f} "
                f"val_weighted_wtr={float(stage2_selected['weighted_wtr']):.3f}"
            )
        if stage_name == "expert_pretrain" and args.save_stage1_checkpoint:
            checkpoint_path = Path(args.save_stage1_checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"saved_stage1_checkpoint={checkpoint_path}")
        if stage_name == "joint_finetune" and global_epoch >= args.min_epochs and stage_stale >= patience:
            break
    for parameter in model.parameters():
        parameter.requires_grad = True
    selected_candidate = None
    if validation_candidates:
        selected_candidate = select_validation_candidate(
            validation_candidates, args.selection_metric, args.selection_mse_tolerance
        )
        best_state = selected_candidate["state"]
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    if args.save_best_checkpoint:
        best_checkpoint_path = Path(args.save_best_checkpoint)
        best_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            best_checkpoint_path,
        )
        print(f"saved_best_checkpoint={best_checkpoint_path}")
    test = evaluate(
        model,
        loaders["test"],
        device,
        data.scaler,
        data.target_idx,
        include_horizon_weights=not args.suppress_horizon_weights,
    )
    horizon_weights = test.pop("horizon_weights", None)
    # Keep selection precision in the log; runner summaries may round for display.
    print(f"Best Val MSE: {best_val:.6f} at epoch {best_epoch}")
    if selected_candidate is not None:
        print(
            f"Selected Val: metric={args.selection_metric} epoch={selected_candidate['epoch']} "
            f"mse={float(selected_candidate['mse']):.6f} "
            f"mae={float(selected_candidate['mae']):.6f} "
            f"weighted_wtr={float(selected_candidate['weighted_wtr']):.3f}"
        )
    print(f"Test: {rounded_metrics(test)}")
    if horizon_weights is not None:
        rounded_horizon_weights = [
            [round(float(weight), 6) for weight in horizon]
            for horizon in horizon_weights
        ]
        print(f"Gate Horizon Weights: {rounded_horizon_weights}")


if __name__ == "__main__":
    main()
