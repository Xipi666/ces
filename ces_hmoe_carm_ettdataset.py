"""CES-HMoE + CARM for causal residual-memory correction on ETT.

CARM stores out-of-fold residuals from the CES-HMoE base model.  At an
inference origin it retrieves only records whose complete future target is
already historical, then applies a learned confidence gate to a clipped
weighted residual correction.  The test split is never added to the bank.

This module reuses the first innovation point implemented in
``ces_hmoe_ettdataset.py`` and adds the second point as an experiment driver.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from ces_hmoe_ettdataset import (
    CESHMoE,
    ETTConfig,
    ETTDataModule,
    ETTWindowDataset,
    combined_loss,
    evaluate,
    rounded_metrics,
    set_seed,
    state_dim,
)


EPS = 1e-8
WORKDAY = 0
WEEKEND = 1
HOLIDAY = 2


def resolve_input_csv(csv_path: Optional[str], data_dir: Optional[str], dataset: str) -> Path:
    """Resolve an explicit CSV or find the standard dataset automatically."""
    if csv_path:
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path

    if data_dir:
        search_dirs = [Path(data_dir).expanduser()]
    else:
        project_dir = Path(__file__).resolve().parent
        search_dirs = [Path.cwd() / "dataset", project_dir / "dataset", Path.cwd(), project_dir]

    candidates = []
    searched = []
    for directory in search_dirs:
        candidate = (directory / f"{dataset}.csv").resolve()
        searched.append(str(candidate))
        if candidate.is_file() and candidate not in candidates:
            candidates.append(candidate)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        locations = ", ".join(str(path) for path in candidates)
        raise FileExistsError(
            f"Multiple CSV files found for {dataset}: {locations}. "
            "Please use --csv or --data_dir to select one."
        )
    searched_paths = "\n  ".join(searched)
    raise FileNotFoundError(
        f"Could not find {dataset}.csv. Searched:\n  {searched_paths}\n"
        "Please use --csv or --data_dir to specify the dataset location."
    )


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / (values.std() + EPS)


def _downsample(values: np.ndarray, points: int) -> np.ndarray:
    """Keep a fixed-size causal summary for keys of different H values."""
    if len(values) == points:
        return values
    positions = np.linspace(0, len(values) - 1, points).round().astype(int)
    return values[positions]


def make_key(
    x: Tensor,
    state: Tensor,
    base: Tensor,
    future_calendar: Tensor,
    key_points: int,
) -> np.ndarray:
    """Build a key only from information available at the forecast origin."""
    history = x.detach().cpu().numpy()
    state_np = state.detach().cpu().numpy()
    base_np = base.detach().cpu().numpy()
    calendar = future_calendar.detach().cpu().numpy()
    keys = []
    for hist_i, state_i, base_i, calendar_i in zip(history, state_np, base_np, calendar):
        hist_summary = _downsample(hist_i, key_points).reshape(-1)
        base_summary = _zscore(_downsample(base_i, key_points))
        cal_summary = np.concatenate([calendar_i.mean(axis=0), calendar_i.std(axis=0)])
        state_summary = np.concatenate([_zscore(state_i), [np.mean(np.abs(state_i))]])
        key = np.concatenate([_zscore(hist_summary), base_summary, cal_summary, state_summary])
        key = key / (np.linalg.norm(key) + EPS)
        keys.append(key.astype(np.float32))
    return np.stack(keys)


@dataclass
class ResidualRecord:
    origin: int
    day_type: int
    key: np.ndarray
    residual: np.ndarray
    base: np.ndarray
    experts: np.ndarray
    state_summary: np.ndarray
    oof_fold: int = -1


class CausalResidualBank:
    def __init__(self, pred_len: int, top_k: int, temperature: float = 0.15) -> None:
        self.pred_len = pred_len
        self.top_k = top_k
        self.temperature = temperature
        self.records: List[ResidualRecord] = []

    def add(self, record: ResidualRecord) -> None:
        if record.residual.shape != (self.pred_len,):
            raise ValueError("Residual shape does not match pred_len")
        self.records.append(record)

    def _candidate_records(self, origin: int, day_type: Optional[int]) -> List[ResidualRecord]:
        return [
            record
            for record in self.records
            if (day_type is None or record.day_type == day_type)
            and record.origin + self.pred_len <= origin
        ]

    def _assert_causal_records(self, origin: int, candidates: Sequence[ResidualRecord]) -> None:
        if any(record.origin + self.pred_len > origin for record in candidates):
            raise AssertionError("Residual bank selected a future-crossing candidate")

    def query(self, origin: int, day_type: int, key: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Retrieve only completed historical futures and return residual stats."""
        candidates = self._candidate_records(origin, day_type)
        self.assert_causal([origin], [candidates])
        if not candidates:
            return np.zeros(self.pred_len, dtype=np.float32), np.ones(self.pred_len, dtype=np.float32), 1e3, 0
        matrix = np.stack([record.key for record in candidates])
        similarity = matrix @ key
        selected = np.argsort(-similarity)[: self.top_k]
        similarity = similarity[selected]
        residuals = np.stack([candidates[index].residual for index in selected])
        weights = np.exp((similarity - similarity.max()) / max(self.temperature, EPS))
        weights = weights / (weights.sum() + EPS)
        mean = (weights[:, None] * residuals).sum(axis=0)
        variance = (weights[:, None] * (residuals - mean) ** 2).sum(axis=0)
        return mean.astype(np.float32), variance.astype(np.float32), float(1.0 - similarity.max()), len(selected)

    def assert_causal(
        self,
        query_origins: Sequence[int],
        candidate_sets: Optional[Sequence[Sequence[ResidualRecord]]] = None,
    ) -> None:
        """Validate candidate lists selected for the supplied query origins."""
        if candidate_sets is None:
            candidate_sets = [self._candidate_records(origin, None) for origin in query_origins]
        if len(candidate_sets) != len(query_origins):
            raise ValueError("candidate_sets must have one entry per query origin")
        for origin, candidates in zip(query_origins, candidate_sets):
            self._assert_causal_records(origin, candidates)


class ConfidenceGate(nn.Module):
    """Horizon-wise alpha in [0, 1] from horizon-wise retrieval features."""

    def __init__(self, pred_len: int, state_features: int = 4, alpha_init: float = 0.02) -> None:
        super().__init__()
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be between 0 and 1")
        self.pred_len = pred_len
        self.input_dim = 4 + state_features
        output = nn.Linear(32, 1)
        # Start from a nearly no-op residual correction.  A zero output weight
        # keeps the initial alpha identical across samples while preserving
        # trainable feature-dependent corrections after the first update.
        nn.init.zeros_(output.weight)
        nn.init.constant_(output.bias, float(np.log(alpha_init / (1.0 - alpha_init))))
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.GELU(),
            output,
        )
        # Each forecast step can learn its own correction strength.
        self.horizon_bias = nn.Parameter(torch.zeros(pred_len))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.pred_len or features.shape[2] != self.input_dim:
            raise ValueError(
                f"Expected gate features [batch, {self.pred_len}, {self.input_dim}], got {tuple(features.shape)}"
            )
        logits = self.net(features).squeeze(-1) + self.horizon_bias[None, :]
        return torch.sigmoid(logits)


def _load_holiday_calendar(
    path: Optional[str],
    years: Set[int],
) -> Tuple[Set[date], Set[date]]:
    if path is not None:
        dates: Set[date] = set()
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            token = line.split(",", 1)[0].strip()
            if not token or token.lower() in {"date", "holiday", "holiday_date"}:
                continue
            dates.add(date.fromisoformat(token[:10]))
        return dates, set()

    try:
        import holidays
        from workalendar.asia import China
    except ImportError as exc:
        raise RuntimeError(
            "Chinese holiday classification requires 'holidays' and 'workalendar'; "
            "install them or pass --holiday_dates."
        ) from exc

    china_holidays = holidays.China(years=sorted(years))
    holiday_dates = {
        value.date() if isinstance(value, datetime) else value
        for value in china_holidays.keys()
    }
    # workalendar exposes known weekend make-up workdays separately. Its
    # current China calendar covers 2018 onward; unsupported years simply have
    # no override and retain the normal weekday/weekend classification.
    calendar = China()
    working_day_overrides = {
        value for value in calendar.extra_working_days if value.year in years
    }
    return holiday_dates, working_day_overrides


def _day_type(data: ETTDataModule, csv_path: str, holiday_dates_path: Optional[str] = None) -> np.ndarray:
    import pandas as pd

    dates = pd.read_csv(csv_path)["date"]
    timestamps = pd.to_datetime(dates)
    holiday_dates, working_day_overrides = _load_holiday_calendar(
        holiday_dates_path,
        set(timestamps.dt.year.astype(int).tolist()),
    )
    return np.asarray(
        [
            HOLIDAY if timestamp.date() in holiday_dates else
            WORKDAY if timestamp.date() in working_day_overrides else
            WEEKEND if timestamp.dayofweek >= 5 else WORKDAY
            for timestamp in timestamps
        ],
        dtype=np.int64,
    )


def _new_model(data: ETTDataModule) -> CESHMoE:
    return CESHMoE(
        data.config.seq_len,
        data.config.pred_len,
        enc_in=len(data.columns),
        entropy_dim=state_dim(len(data.columns), len(data.config.entropy_scales)),
    )


def fit_base(
    data: ETTDataModule,
    indices: Sequence[int],
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_epochs: int,
    balance_weight: float,
    device: str,
    validation_loader: Optional[DataLoader] = None,
) -> Tuple[CESHMoE, int]:
    dataset = data.dataset("train")
    train_indices = list(indices)
    if validation_loader is None:
        # OOF models validate on a trailing slice of their causal training prefix.
        # The slice is removed from fitting, so its labels never train the model.
        if len(train_indices) < 2:
            raise ValueError("At least two training windows are required for validation-based selection")
        validation_size = max(1, len(train_indices) // 5)
        validation_indices = train_indices[-validation_size:]
        train_indices = train_indices[:-validation_size]
        validation_loader = DataLoader(
            Subset(dataset, validation_indices),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=data.config.pin_memory,
        )
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=data.config.pin_memory,
    )
    model = _new_model(data).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
        threshold=1e-4, min_lr=1e-6
    )
    best_state: Optional[Dict[str, Tensor]] = None
    best_val = float("inf")
    stale = 0
    completed_epochs = 0
    for epoch in range(1, epochs + 1):
        completed_epochs = epoch
        model.train()
        for batch in loader:
            non_blocking = device.startswith("cuda")
            x = batch["x"].to(device, non_blocking=non_blocking).float()
            future = batch["future_calendar"].to(device, non_blocking=non_blocking).float()
            y = batch["y"].to(device, non_blocking=non_blocking).float()
            state = batch["state"].to(device, non_blocking=non_blocking).float()
            output = model(x, future, state)
            loss = combined_loss(output["prediction"], y, output["weights"], data.ramp_threshold, balance_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = evaluate(model, validation_loader, device)
        current = validation["mse"]
        scheduler.step(current)
        if current < best_val:
            best_val = current
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if epoch >= min_epochs and stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, completed_epochs


@torch.no_grad()
def collect_records(
    model: CESHMoE,
    dataset: ETTWindowDataset,
    indices: Sequence[int],
    data: ETTDataModule,
    day_types: np.ndarray,
    key_points: int,
    device: str,
    oof_fold: int = -1,
) -> List[ResidualRecord]:
    model.eval()
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=data.config.batch_size,
        shuffle=False,
        pin_memory=data.config.pin_memory,
    )
    records: List[ResidualRecord] = []
    offset = 0
    non_blocking = device.startswith("cuda")
    for batch in loader:
        x = batch["x"].to(device, non_blocking=non_blocking).float()
        future = batch["future_calendar"].to(device, non_blocking=non_blocking).float()
        state = batch["state"].to(device, non_blocking=non_blocking).float()
        output = model(x, future, state)
        base = output["prediction"]
        keys = make_key(x, state, base, future, key_points)
        y = batch["y"].cpu().numpy().astype(np.float32)
        base_np = base.cpu().numpy().astype(np.float32)
        experts = output["experts"].cpu().numpy().astype(np.float32)
        for row in range(len(y)):
            sample_index = indices[offset + row]
            origin = dataset.origins[sample_index] + dataset.seq_len
            records.append(ResidualRecord(
                origin=origin,
                day_type=int(day_types[origin]),
                key=keys[row],
                residual=y[row] - base_np[row],
                base=base_np[row],
                experts=experts[row],
                state_summary=_downsample(state[row].cpu().numpy(), 4),
                oof_fold=oof_fold,
            ))
        offset += len(y)
    return records


def build_oof_bank(
    data: ETTDataModule,
    day_types: np.ndarray,
    folds: int,
    oof_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    min_epochs: int,
    balance_weight: float,
    key_points: int,
    device: str,
) -> Tuple[CausalResidualBank, List[ResidualRecord]]:
    dataset = data.dataset("train")
    total = len(dataset)
    if folds < 1 or total < folds + 1:
        raise ValueError("Not enough training windows for the requested OOF folds")
    bank = CausalResidualBank(data.config.pred_len, top_k=8)
    records: List[ResidualRecord] = []
    boundaries = np.linspace(0, total, folds + 2, dtype=int)
    for fold in range(folds):
        train_end = boundaries[fold + 1]
        valid_start, valid_end = boundaries[fold + 1], boundaries[fold + 2]
        if train_end < 1 or valid_start >= valid_end:
            continue
        model, _ = fit_base(data, range(train_end), oof_epochs, batch_size, lr, patience, min(min_epochs, oof_epochs), balance_weight, device)
        records.extend(
            collect_records(
                model,
                dataset,
                range(valid_start, valid_end),
                data,
                day_types,
                key_points,
                device,
                oof_fold=fold,
            )
        )
        print(f"OOF fold {fold + 1}/{folds}: train={train_end} holdout={valid_end - valid_start}")
    records.sort(key=lambda record: record.origin)
    for record in records:
        bank.add(record)
    return bank, records


def retrieval_features(
    record: ResidualRecord,
    bank: CausalResidualBank,
    state: np.ndarray,
    query_origin: int,
    day_type: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    correction, variance, distance, count = bank.query(query_origin, day_type, record.key)
    disagreement = np.std(record.experts, axis=-1)
    state_features = _zscore(_downsample(state, 4))
    horizon = bank.pred_len
    features = np.concatenate(
        [
            np.full((horizon, 1), distance, dtype=np.float32),
            np.log(variance + EPS).reshape(horizon, 1).astype(np.float32),
            np.broadcast_to(state_features, (horizon, len(state_features))).astype(np.float32),
            disagreement.reshape(horizon, 1).astype(np.float32),
            np.full((horizon, 1), float(count) / max(bank.top_k, 1), dtype=np.float32),
        ],
        axis=1,
    )
    return features, correction, variance, distance, count


def _gate_samples(
    records: Sequence[ResidualRecord],
    bank: CausalResidualBank,
    residual_low: np.ndarray,
    residual_high: np.ndarray,
    initial_records: Sequence[ResidualRecord] = (),
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Build causal gate samples, optionally starting from past records."""
    historical = CausalResidualBank(bank.pred_len, bank.top_k, bank.temperature)
    for record in sorted(initial_records, key=lambda item: item.origin):
        historical.add(record)

    features, corrections, targets, bases = [], [], [], []
    for record in sorted(records, key=lambda item: item.origin):
        feature, correction, _, _, count = retrieval_features(
            record,
            historical,
            record.state_summary,
            record.origin,
            record.day_type,
        )
        if count:
            features.append(feature)
            corrections.append(np.clip(correction, residual_low, residual_high))
            targets.append(record.base + record.residual)
            bases.append(record.base)
        historical.add(record)

    if not features:
        return None
    return (
        np.stack(features).astype(np.float32),
        np.stack(corrections).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.stack(bases).astype(np.float32),
    )


def _gate_metrics(
    gate: ConfidenceGate,
    samples: Tuple[Tensor, Tensor, Tensor, Tensor],
    alpha_sparsity_weight: float,
) -> Dict[str, float]:
    x, correction, target, base = samples
    alpha = gate(x)
    prediction = base + alpha * correction
    fusion_loss = F.smooth_l1_loss(prediction, target)
    sparsity_loss = alpha.abs().mean()
    return {
        "loss": float((fusion_loss + alpha_sparsity_weight * sparsity_loss).item()),
        "fusion_loss": float(fusion_loss.item()),
        "mse": float(F.mse_loss(prediction, target).item()),
        "base_mse": float(F.mse_loss(base, target).item()),
        "alpha_mean": float(alpha.mean().item()),
    }


def _tensor_gate_samples(
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    device: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return tuple(torch.from_numpy(array).to(device) for array in arrays)  # type: ignore[return-value]


def _gate_diagnosis(
    history: Sequence[Dict[str, float]],
    best_epoch: int,
    max_epochs: int,
) -> str:
    if not history:
        return "NO_SAMPLES"
    best = history[best_epoch - 1]
    last = history[-1]
    if best_epoch >= max_epochs and last["mse"] <= best["mse"] * (1.0 + 1e-6):
        return "UNDERFIT_SUSPECTED(best_epoch_at_max_epochs)"
    if (
        last["epoch"] > best_epoch
        and last["mse"] > best["mse"]
        and last["train_mse"] <= best["train_mse"] * (1.0 + 1e-3)
    ):
        return f"OVERFIT_SIGNAL(val_worse_after_epoch_{best_epoch})"
    return "NO_CLEAR_SIGNAL"


def _train_gate(
    gate: ConfidenceGate,
    train_samples: Tuple[Tensor, Tensor, Tensor, Tensor],
    epochs: int,
    learning_rate: float,
    alpha_sparsity_weight: float,
    validation_samples: Optional[Tuple[Tensor, Tensor, Tensor, Tensor]] = None,
    patience: int = 15,
    min_epochs: int = 15,
    select_best: bool = False,
    log_prefix: str = "Gate",
) -> Tuple[ConfidenceGate, int, List[Dict[str, float]]]:
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)
    best_state: Optional[Dict[str, Tensor]] = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        gate.train()
        x, correction, target, base = train_samples
        alpha = gate(x)
        prediction = base + alpha * correction
        fusion_loss = F.smooth_l1_loss(prediction, target)
        sparsity_loss = alpha.abs().mean()
        loss = fusion_loss + alpha_sparsity_weight * sparsity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        gate.eval()
        with torch.no_grad():
            train_metrics = _gate_metrics(gate, train_samples, alpha_sparsity_weight)
            validation_metrics = (
                _gate_metrics(gate, validation_samples, alpha_sparsity_weight)
                if validation_samples is not None
                else None
            )

        current = validation_metrics["mse"] if validation_metrics is not None else train_metrics["mse"]
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "fusion_loss": train_metrics["fusion_loss"],
            "mse": current,
            "train_mse": train_metrics["mse"],
            "alpha_mean": train_metrics["alpha_mean"],
        }
        if validation_metrics is not None:
            row["val_loss"] = validation_metrics["loss"]
            row["val_mse"] = validation_metrics["mse"]
            row["val_alpha_mean"] = validation_metrics["alpha_mean"]
        history.append(row)

        improved = current < best_val
        if improved:
            best_val = current
            best_epoch = epoch
            if select_best:
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in gate.state_dict().items()
                }
            stale = 0
        else:
            stale += 1

        status = "improving" if improved else "val_worse" if validation_metrics is not None and current > best_val else "plateau"
        if epoch == 1 or epoch % 10 == 0 or improved or (select_best and stale >= patience):
            if validation_metrics is None:
                print(
                    f"{log_prefix} epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.6f} "
                    f"train_mse={train_metrics['mse']:.6f} alpha_mean={train_metrics['alpha_mean']:.6f}",
                    flush=True,
                )
            else:
                print(
                    f"{log_prefix} epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.6f} "
                    f"train_mse={train_metrics['mse']:.6f} val_loss={validation_metrics['loss']:.6f} "
                    f"val_mse={validation_metrics['mse']:.6f} best_val_mse={best_val:.6f} "
                    f"alpha_mean={train_metrics['alpha_mean']:.6f} status={status}",
                    flush=True,
                )

        if select_best and epoch >= min_epochs and stale >= patience:
            print(f"{log_prefix} early stopping at epoch {epoch} (best_epoch={best_epoch})", flush=True)
            break

    if select_best and best_state is not None:
        gate.load_state_dict(best_state)
    return gate, best_epoch, history


def fit_confidence_gate(
    records: Sequence[ResidualRecord],
    bank: CausalResidualBank,
    data: ETTDataModule,
    day_types: np.ndarray,
    residual_low: np.ndarray,
    residual_high: np.ndarray,
    device: str,
    epochs: int = 80,
    learning_rate: float = 1e-4,
    alpha_sparsity_weight: float = 0.01,
    alpha_init: float = 0.02,
    patience: int = 15,
    min_epochs: int = 15,
) -> Tuple[ConfidenceGate, int]:
    """Select gate epochs chronologically, then refit on every OOF record."""
    if alpha_sparsity_weight < 0.0:
        raise ValueError("alpha_sparsity_weight must be non-negative")
    if epochs < 1 or patience < 1 or min_epochs < 1:
        raise ValueError("Gate epochs, patience and min_epochs must be positive")
    if min_epochs > epochs:
        raise ValueError("Gate min_epochs cannot be greater than gate epochs")

    train_records = sorted(
        (record for record in records if 0 <= record.oof_fold < 4),
        key=lambda record: record.origin,
    )
    validation_records = sorted(
        (record for record in records if record.oof_fold == 4),
        key=lambda record: record.origin,
    )
    if not train_records or not validation_records:
        raise ValueError("Gate selection requires non-empty OOF 1-4 training and OOF 5 validation records")

    # Bounds for model selection are estimated from OOF 1-4 only. OOF 5 is
    # kept entirely out of preprocessing and is used only for validation.
    train_residuals = np.stack([record.residual for record in train_records])
    train_low = np.quantile(train_residuals, 0.05, axis=0).astype(np.float32)
    train_high = np.quantile(train_residuals, 0.95, axis=0).astype(np.float32)
    train_arrays = _gate_samples(train_records, bank, train_low, train_high)
    validation_arrays = _gate_samples(
        validation_records,
        bank,
        train_low,
        train_high,
        initial_records=train_records,
    )
    if train_arrays is None or validation_arrays is None:
        raise ValueError("Gate selection requires causal samples in both OOF 1-4 and OOF 5")
    train_samples = _tensor_gate_samples(train_arrays, device)
    validation_samples = _tensor_gate_samples(validation_arrays, device)
    print(
        f"Gate selection split: train_oof_records={len(train_records)} "
        f"val_oof_records={len(validation_records)} train_samples={len(train_arrays[0])} "
        f"val_samples={len(validation_arrays[0])}",
        flush=True,
    )

    selection_gate = ConfidenceGate(bank.pred_len, alpha_init=alpha_init).to(device)
    _, best_epoch, history = _train_gate(
        selection_gate,
        train_samples,
        epochs,
        learning_rate,
        alpha_sparsity_weight,
        validation_samples=validation_samples,
        patience=patience,
        min_epochs=min_epochs,
        select_best=True,
        log_prefix="Gate select",
    )
    diagnosis = _gate_diagnosis(history, best_epoch, epochs)
    best_val = history[best_epoch - 1]["val_mse"]
    last_val = history[-1]["val_mse"]
    print(
        f"Gate selection result: best_epoch={best_epoch} best_val_mse={best_val:.6f} "
        f"last_val_mse={last_val:.6f} diagnosis={diagnosis}",
        flush=True,
    )

    # After epoch selection, refit from scratch on every OOF record. The
    # validation fold is now legal training data because selection is done.
    all_records = sorted(records, key=lambda record: record.origin)
    all_residuals = np.stack([record.residual for record in all_records])
    all_low = np.quantile(all_residuals, 0.05, axis=0).astype(np.float32)
    all_high = np.quantile(all_residuals, 0.95, axis=0).astype(np.float32)
    final_arrays = _gate_samples(all_records, bank, all_low, all_high)
    if final_arrays is None:
        raise ValueError("Gate refit requires causal samples across all OOF records")
    final_samples = _tensor_gate_samples(final_arrays, device)
    print(
        f"Gate refit: all_oof_records={len(all_records)} samples={len(final_arrays[0])} "
        f"epochs={best_epoch}",
        flush=True,
    )
    final_gate = ConfidenceGate(bank.pred_len, alpha_init=alpha_init).to(device)
    final_gate, _, final_history = _train_gate(
        final_gate,
        final_samples,
        best_epoch,
        learning_rate,
        alpha_sparsity_weight,
        log_prefix="Gate refit",
    )
    final_metrics = final_history[-1]
    print(
        f"Gate refit result: train_mse={final_metrics['train_mse']:.6f} "
        f"train_loss={final_metrics['train_loss']:.6f} alpha_mean={final_metrics['alpha_mean']:.6f} "
        f"selection_diagnosis={diagnosis}",
        flush=True,
    )
    return final_gate, best_epoch


@torch.no_grad()
def evaluate_carm(
    model: CESHMoE,
    gate: ConfidenceGate,
    bank: CausalResidualBank,
    loader: DataLoader,
    dataset: ETTWindowDataset,
    day_types: np.ndarray,
    residual_low: np.ndarray,
    residual_high: np.ndarray,
    key_points: int,
    device: str,
    use_carm: bool = True,
) -> Dict[str, float]:
    model.eval()
    gate.eval()
    base_errors, final_errors = [], []
    expert_weights = []
    offset = 0
    for batch in loader:
        non_blocking = device.startswith("cuda")
        x = batch["x"].to(device, non_blocking=non_blocking).float()
        future = batch["future_calendar"].to(device, non_blocking=non_blocking).float()
        state = batch["state"].to(device, non_blocking=non_blocking).float()
        output = model(x, future, state)
        base = output["prediction"]
        expert_weights.append(output["weights"].cpu().numpy())
        keys = make_key(x, state, base, future, key_points=key_points)
        base_np = base.cpu().numpy()
        experts = output["experts"].cpu().numpy()
        corrected = []
        gate_features = []
        for row in range(len(x)):
            origin = dataset.origins[offset + row] + dataset.seq_len
            record = ResidualRecord(origin, int(day_types[origin]), keys[row], np.zeros(bank.pred_len), base_np[row], experts[row], _downsample(state[row].cpu().numpy(), 4))
            feature, correction, _, _, _ = retrieval_features(record, bank, record.state_summary, origin, record.day_type)
            gate_features.append(feature)
            corrected.append(np.clip(correction, residual_low, residual_high))
        if use_carm:
            alpha = gate(torch.from_numpy(np.stack(gate_features)).to(device)).cpu().numpy()
            prediction = base_np + alpha * np.stack(corrected)
        else:
            # Validation-selected no-op path: alpha is exactly zero for every horizon.
            prediction = base_np
        target = batch["y"].numpy()
        base_errors.append(torch.from_numpy(base_np - target))
        final_errors.append(torch.from_numpy(prediction - target))
        offset += len(x)
    base_error = torch.cat(base_errors)
    final_error = torch.cat(final_errors)
    weights = np.concatenate(expert_weights, axis=0)
    dominant = np.argmax(weights, axis=-1)
    metrics = {
        "base_mse": float((base_error ** 2).mean()),
        "base_mae": float(base_error.abs().mean()),
        "mse": float((final_error ** 2).mean()),
        "mae": float(final_error.abs().mean()),
        "expert_trend_weight": float(weights[..., 0].mean()),
        "expert_periodic_weight": float(weights[..., 1].mean()),
        "expert_ramp_weight": float(weights[..., 2].mean()),
        "expert_trend_dominant_ratio": float((dominant == 0).mean()),
        "expert_periodic_dominant_ratio": float((dominant == 1).mean()),
        "expert_ramp_dominant_ratio": float((dominant == 2).mean()),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CES-HMoE with causal OOF residual memory (CARM).")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--data_dir", default=None, help="Optional dataset directory; auto-detected when omitted.")
    parser.add_argument("--dataset", default="ETTh1", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--oof_epochs", type=int, default=100)
    parser.add_argument("--oof_folds", type=int, default=5)
    parser.add_argument("--gate_epochs", type=int, default=80)
    parser.add_argument("--gate_patience", type=int, default=15)
    parser.add_argument("--gate_min_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gate_lr", type=float, default=1e-4)
    parser.add_argument("--alpha_sparsity_weight", type=float, default=0.01)
    parser.add_argument("--alpha_init", type=float, default=0.02)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_epochs", type=int, default=15)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--key_points", type=int, default=32)
    parser.add_argument(
        "--holiday_dates",
        default=None,
        help="Optional file with one ISO date per line; replaces the built-in Chinese holiday calendar.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--balance_weight", type=float, default=0.01)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        print("CARM smoke test requires torch and a small synthetic data module; CLI parsing is valid.")
        return
    if min(
        args.epochs,
        args.oof_epochs,
        args.oof_folds,
        args.gate_epochs,
        args.gate_patience,
        args.gate_min_epochs,
        args.batch_size,
        args.patience,
        args.min_epochs,
        args.top_k,
        args.key_points,
    ) < 1:
        parser.error("training, fold, batch, top_k and key_points arguments must be positive")
    if args.lr <= 0 or args.gate_lr <= 0:
        parser.error("--lr and --gate_lr must be positive")
    if args.alpha_sparsity_weight < 0:
        parser.error("--alpha_sparsity_weight must be non-negative")
    if not 0 < args.alpha_init < 1:
        parser.error("--alpha_init must be between 0 and 1")
    if args.min_epochs > args.epochs:
        parser.error("--min_epochs cannot be greater than --epochs")
    if args.oof_folds != 5:
        parser.error("--oof_folds must be 5 for OOF 1-4 gate training and OOF 5 validation")
    if args.gate_min_epochs > args.gate_epochs:
        parser.error("--gate_min_epochs cannot be greater than --gate_epochs")
    set_seed(args.seed)
    csv_path = resolve_input_csv(args.csv, args.data_dir, args.dataset)
    config = ETTConfig(str(csv_path), dataset=args.dataset, seq_len=args.seq_len, pred_len=args.pred_len, batch_size=args.batch_size)
    data = ETTDataModule(config)
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    if args.device == "cuda" and device != "cuda":
        parser.error("CUDA requested but unavailable")
    day_types = _day_type(data, str(csv_path), args.holiday_dates)
    bank, oof_records = build_oof_bank(data, day_types, args.oof_folds, args.oof_epochs, args.batch_size, args.lr, args.patience, args.min_epochs, args.balance_weight, args.key_points, device)
    bank.top_k = args.top_k
    residuals = np.stack([record.residual for record in oof_records])
    residual_low = np.quantile(residuals, 0.05, axis=0).astype(np.float32)
    residual_high = np.quantile(residuals, 0.95, axis=0).astype(np.float32)
    gate, gate_best_epoch = fit_confidence_gate(
        oof_records, bank, data, day_types, residual_low, residual_high, device,
        epochs=args.gate_epochs,
        learning_rate=args.gate_lr,
        alpha_sparsity_weight=args.alpha_sparsity_weight,
        alpha_init=args.alpha_init,
        patience=args.gate_patience,
        min_epochs=args.gate_min_epochs,
    )
    # OOF training consumes RNG state. Reset it so the final base model is
    # reproducible and uses the same initialization as standalone CES-HMoE.
    set_seed(args.seed)
    loaders = data.loaders()
    train_dataset = data.dataset("train")
    final_model, base_epochs = fit_base(
        data,
        range(len(train_dataset)),
        args.epochs,
        args.batch_size,
        args.lr,
        args.patience,
        args.min_epochs,
        args.balance_weight,
        device,
        validation_loader=loaders["val"],
    )
    print(
        f"dataset={args.dataset} csv={csv_path} seq_len={config.seq_len} "
        f"pred_len={config.pred_len} epochs={args.epochs} lr={args.lr:.2e} "
        f"gate_lr={args.gate_lr:.2e} device={device} "
        f"alpha_init={args.alpha_init:.4f} alpha_sparsity_weight={args.alpha_sparsity_weight:.2e} "
        f"oof_records={len(oof_records)} bank_top_k={args.top_k} "
        f"gate_best_epoch={gate_best_epoch}"
    )
    print(f"Base training epochs: {base_epochs}/{args.epochs}")
    val_candidate = evaluate_carm(
        final_model, gate, bank, loaders["val"], data.dataset("val"),
        day_types, residual_low, residual_high, args.key_points, device,
        use_carm=True,
    )
    # Select the correction only with validation data. Test data never controls
    # whether the residual memory is enabled.
    use_carm = val_candidate["mse"] < val_candidate["base_mse"]
    val = val_candidate if use_carm else evaluate_carm(
        final_model, gate, bank, loaders["val"], data.dataset("val"),
        day_types, residual_low, residual_high, args.key_points, device,
        use_carm=False,
    )
    test = evaluate_carm(
        final_model, gate, bank, loaders["test"], data.dataset("test"),
        day_types, residual_low, residual_high, args.key_points, device,
        use_carm=use_carm,
    )
    mode = "CARM" if use_carm else "BASE(alpha=0)"
    print(f"CARM Val Candidate: {rounded_metrics(val_candidate)}")
    print(f"Selected mode: {mode}")
    print(f"Selected Val: {rounded_metrics(val)}")
    print(f"Selected Test: {rounded_metrics(test)}")
    print(
        "Test expert usage: "
        f"trend_weight={test['expert_trend_weight']:.3f} "
        f"periodic_weight={test['expert_periodic_weight']:.3f} "
        f"ramp_weight={test['expert_ramp_weight']:.3f} "
        f"trend_dominant={test['expert_trend_dominant_ratio']:.3f} "
        f"periodic_dominant={test['expert_periodic_dominant_ratio']:.3f} "
        f"ramp_dominant={test['expert_ramp_dominant_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
