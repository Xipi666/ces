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
        return {
            split: DataLoader(
                self.dataset(split),
                batch_size=self.config.batch_size,
                shuffle=split == "train",
                num_workers=self.config.num_workers,
                drop_last=split == "train",
                pin_memory=self.config.pin_memory,
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

    def __init__(self, seq_len: int, pred_len: int) -> None:
        super().__init__()
        kernel = min(25, seq_len if seq_len % 2 else seq_len - 1)
        self.avg = nn.AvgPool1d(kernel, stride=1, padding=kernel // 2, count_include_pad=False)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        target = x[..., -1:].transpose(1, 2)
        trend = self.avg(target).squeeze(1)
        return self.linear(trend)


class PeriodicExpert(nn.Module):
    """Small convolutional periodic expert with a direct H-vector head."""

    def __init__(self, seq_len: int, pred_len: int, hidden: int = 32) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, hidden, kernel_size=25, padding=12),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, groups=hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Linear(hidden * 8, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        target = x[..., -1:].transpose(1, 2)
        return self.head(self.conv(target).flatten(1))


class RampExpert(nn.Module):
    """Dilated TCN expert using all historical channels and first differences."""

    def __init__(self, enc_in: int, pred_len: int, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(enc_in, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, pred_len)

    def forward(self, x: Tensor) -> Tensor:
        diff = torch.diff(x, dim=1, prepend=x[:, :1])
        hidden = self.net(diff.transpose(1, 2))[:, :, -1]
        return self.head(hidden)


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
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.temperature = temperature
        self.history = HistoryEncoder(enc_in, route_dim)
        self.state = nn.Sequential(nn.LayerNorm(entropy_dim), nn.Linear(entropy_dim, route_dim), nn.GELU())
        self.future = nn.Sequential(nn.Linear(calendar_dim, route_dim), nn.GELU())
        self.horizon = nn.Parameter(torch.randn(pred_len, route_dim) * 0.02)
        self.trend = TrendExpert(seq_len, pred_len)
        self.periodic = PeriodicExpert(seq_len, pred_len)
        self.ramp = RampExpert(enc_in, pred_len)
        self.gate = nn.Sequential(
            nn.Linear(route_dim * 3, route_dim),
            nn.GELU(),
            nn.Linear(route_dim, 3),
        )

    def forward(self, x: Tensor, future_calendar: Tensor, state: Tensor) -> Dict[str, Tensor]:
        hist = self.history(x)
        state_vec = self.state(state)
        future = self.future(future_calendar)
        route = torch.cat(
            [
                hist[:, None, :].expand(-1, self.pred_len, -1),
                state_vec[:, None, :].expand(-1, self.pred_len, -1),
                future + self.horizon[None, :, :],
            ],
            dim=-1,
        )
        logits = self.gate(route)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        expert_predictions = torch.stack(
            [self.trend(x), self.periodic(x), self.ramp(x)], dim=-1
        )
        prediction = (weights * expert_predictions).sum(dim=-1)
        return {"prediction": prediction, "weights": weights, "experts": expert_predictions}


def combined_loss(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    ramp_quantile: float,
    balance_weight: float = 0.01,
) -> Tensor:
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
    usage = weights.mean(dim=(0, 1))
    balance = ((usage - 1.0 / 3.0) ** 2).sum()
    return base + 0.2 * slope + 0.3 * ramp + balance_weight * balance


def train_one_epoch(
    model: CESHMoE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    ramp_quantile: float,
    balance_weight: float,
    fusion: str = "gate",
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
        else:
            prediction = output["prediction"]
            weights = output["weights"]
        loss = combined_loss(
            prediction,
            y,
            weights,
            ramp_quantile=ramp_quantile,
            balance_weight=balance_weight,
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
) -> Dict[str, float]:
    model.eval()
    errors = []
    absolute = []
    expert_weights = []
    non_blocking = device.startswith("cuda")
    for batch in loader:
        output = model(
            batch["x"].to(device, non_blocking=non_blocking).float(),
            batch["future_calendar"].to(device, non_blocking=non_blocking).float(),
            batch["state"].to(device, non_blocking=non_blocking).float(),
        )
        expert_weights.append(output["weights"].cpu())
        error = output["prediction"] - batch["y"].to(device, non_blocking=non_blocking).float()
        errors.append(error.cpu())
        absolute.append(error.abs().cpu())
    error = torch.cat(errors)
    absolute = torch.cat(absolute)
    weights = torch.cat(expert_weights)
    dominant = weights.argmax(dim=-1)
    metrics = {
        "mse": float((error**2).mean()),
        "rmse": float(torch.sqrt((error**2).mean())),
        "mae": float(absolute.mean()),
        "trend_weight": float(weights[..., 0].mean()),
        "periodic_weight": float(weights[..., 1].mean()),
        "ramp_weight": float(weights[..., 2].mean()),
        "trend_dominant": float((dominant == 0).float().mean()),
        "periodic_dominant": float((dominant == 1).float().mean()),
        "ramp_dominant": float((dominant == 2).float().mean()),
    }
    if scaler is not None:
        if target_idx is None:
            raise ValueError("target_idx is required when reporting original-scale metrics")
        raw_error = error * float(scaler.std[target_idx])
        metrics.update(
            {
                "mse_raw": float((raw_error**2).mean()),
                "rmse_raw": float(torch.sqrt((raw_error**2).mean())),
                "mae_raw": float(raw_error.abs().mean()),
            }
        )
    return metrics


def smoke_test() -> None:
    """Check all tensor contracts without requiring a downloaded ETT CSV."""
    batch, seq_len, pred_len, channels, entropy_dim = 4, 168, 24, 7, state_dim(7, 3)
    model = CESHMoE(seq_len, pred_len, channels, entropy_dim)
    output = model(
        torch.randn(batch, seq_len, channels),
        torch.randn(batch, pred_len, 6),
        torch.randn(batch, entropy_dim),
    )
    assert output["prediction"].shape == (batch, pred_len)
    assert output["weights"].shape == (batch, pred_len, 3)
    assert output["experts"].shape == (batch, pred_len, 3)
    assert torch.allclose(output["weights"].sum(-1), torch.ones(batch, pred_len), atol=1e-5)
    loss = combined_loss(output["prediction"], torch.randn(batch, pred_len), output["weights"], 1.0)
    loss.backward()
    print("smoke test passed", {k: tuple(v.shape) for k, v in output.items()})


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--data_dir", default="./dataset")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--k", type=int, default=2, help="Reserved for the optional sparse router.")
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument(
        "--stage1_epochs", type=int, default=None,
        help="Expert pretraining epochs; defaults to 30%% of --epochs.",
    )
    parser.add_argument(
        "--stage2_epochs", type=int, default=None,
        help="Frozen-expert gate training epochs; defaults to 30%% of --epochs.",
    )
    parser.add_argument(
        "--finetune_lr", type=float, default=None,
        help="Stage-3 joint fine-tuning learning rate; defaults to lr/10.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        smoke_test()
        return
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.min_epochs < 1:
        parser.error("--epochs, --batch_size, --patience and --min_epochs must be positive")
    if args.min_epochs > args.epochs:
        parser.error("--min_epochs cannot be greater than --epochs")
    stage1_epochs = args.stage1_epochs if args.stage1_epochs is not None else max(1, args.epochs * 3 // 10)
    stage2_epochs = args.stage2_epochs if args.stage2_epochs is not None else max(1, args.epochs * 3 // 10)
    stage3_epochs = args.epochs - stage1_epochs - stage2_epochs
    if stage1_epochs < 1 or stage2_epochs < 1 or stage3_epochs < 1:
        parser.error("stage1_epochs + stage2_epochs must be less than epochs")
    if args.finetune_lr is not None and args.finetune_lr <= 0:
        parser.error("--finetune_lr must be positive")
    set_seed(args.seed)
    csv_path = Path(args.csv) if args.csv else resolve_csv(args.data_dir, args.dataset)
    config = ETTConfig(
        str(csv_path),
        dataset=args.dataset,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
    )
    data = ETTDataModule(config)
    loaders = data.loaders()
    model = CESHMoE(
        config.seq_len,
        config.pred_len,
        enc_in=len(data.columns),
        entropy_dim=state_dim(len(data.columns), len(config.entropy_scales)),
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    model.to(device)
    stage3_lr = args.finetune_lr if args.finetune_lr is not None else args.lr / 10.0
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    print(f"dataset={args.dataset} csv={csv_path} seq_len={config.seq_len} pred_len={config.pred_len} device={device} seed={args.seed}")
    print(
        f"staged_training=expert_pretrain:{stage1_epochs} "
        f"gate_frozen_experts:{stage2_epochs} joint_finetune:{stage3_epochs} "
        f"finetune_lr={stage3_lr:.2e}"
    )
    stages = [
        ("expert_pretrain", stage1_epochs, args.lr, "uniform", True),
        ("gate_frozen_experts", stage2_epochs, args.lr, "gate", False),
        ("joint_finetune", stage3_epochs, stage3_lr, "gate", True),
    ]
    global_epoch = 0
    for stage_name, stage_epochs, stage_lr, fusion, train_experts in stages:
        for parameter in model.trend.parameters():
            parameter.requires_grad = train_experts
        for parameter in model.periodic.parameters():
            parameter.requires_grad = train_experts
        for parameter in model.ramp.parameters():
            parameter.requires_grad = train_experts
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
        if stage_name == "joint_finetune":
            stale = 0
        for stage_epoch in range(1, stage_epochs + 1):
            global_epoch += 1
            loss = train_one_epoch(
                model, loaders["train"], optimizer, device,
                data.ramp_threshold, args.balance_weight, fusion=fusion,
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
            if stage_name != "expert_pretrain" and val["mse"] < best_val:
                best_val = val["mse"]
                best_epoch = global_epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stage_name == "joint_finetune" and global_epoch >= args.min_epochs and stale >= args.patience:
                print(f"Early stopping at epoch {global_epoch}")
                break
        if stage_name == "joint_finetune" and global_epoch >= args.min_epochs and stale >= args.patience:
            break
    for parameter in model.parameters():
        parameter.requires_grad = True
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    test = evaluate(model, loaders["test"], device, data.scaler, data.target_idx)
    print(f"Best Val MSE: {best_val:.3f} at epoch {best_epoch}")
    print(f"Test: {rounded_metrics(test)}")


if __name__ == "__main__":
    main()
