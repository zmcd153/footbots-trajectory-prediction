from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PitchSize:
    width: float = 105.0
    height: float = 68.0


class SetAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention over the players in one frame."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        return self.norm2(x + self.dropout(self.ffn(x)))


class PlayerSetEncoder(nn.Module):
    """Set Transformer style encoder for multi-agent context at each frame."""

    def __init__(
        self,
        input_dim: int = 6,
        dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([SetAttentionBlock(dim, heads, dropout) for _ in range(depth)])
        self.out = nn.LayerNorm(dim)

    def forward(self, players: torch.Tensor) -> torch.Tensor:
        batch, time, agents, _ = players.shape
        visible = players[..., -1] > 0.5
        x = self.input_proj(players).reshape(batch * time, agents, -1)
        pad_mask = (~visible).reshape(batch * time, agents)

        # MultiheadAttention cannot handle rows where every key is masked.
        all_missing = pad_mask.all(dim=1)
        if all_missing.any():
            pad_mask = pad_mask.clone()
            pad_mask[all_missing, 0] = False

        for block in self.blocks:
            x = block(x, key_padding_mask=pad_mask)

        valid = (~pad_mask).unsqueeze(-1).to(x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return self.out(pooled).reshape(batch, time, -1)


class BallTrajectorySetBiLSTM(nn.Module):
    """Ball trajectory inference from player context and ball history.

    This follows the paper idea at code level:
    1. Set Transformer encodes unordered players in each frame.
    2. A Bi-LSTM models the temporal evolution of player context.
    3. A second Bi-LSTM models the observed ball sequence.
    4. A decoder predicts future ball displacements.
    """

    def __init__(
        self,
        obs_steps: int = 20,
        pred_steps: int = 40,
        player_input_dim: int = 6,
        ball_input_dim: int = 5,
        dim: int = 128,
        set_depth: int = 2,
        lstm_layers: int = 1,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.player_encoder = PlayerSetEncoder(player_input_dim, dim, set_depth, heads, dropout)
        self.ball_proj = nn.Sequential(nn.Linear(ball_input_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.context_lstm = nn.LSTM(dim, dim // 2, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.ball_lstm = nn.LSTM(dim, dim // 2, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.context_fuse = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.LayerNorm(dim))
        self.step_embed = nn.Embedding(pred_steps, dim)
        self.decoder = nn.LSTM(dim, dim, batch_first=True)
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, players: torch.Tensor, ball_obs: torch.Tensor) -> torch.Tensor:
        if players.shape[1] != self.obs_steps or ball_obs.shape[1] != self.obs_steps:
            raise ValueError(f"Expected {self.obs_steps} observed steps.")

        player_seq = self.player_encoder(players)
        ball_seq = self.ball_proj(ball_obs)
        context_seq, _ = self.context_lstm(player_seq)
        ball_seq, _ = self.ball_lstm(ball_seq)

        context = self.context_fuse(torch.cat([context_seq[:, -1], ball_seq[:, -1]], dim=-1))
        steps = torch.arange(self.pred_steps, device=players.device)
        decoder_in = self.step_embed(steps).unsqueeze(0).expand(players.shape[0], -1, -1)
        decoder_in = decoder_in + context.unsqueeze(1)
        decoded, _ = self.decoder(decoder_in)

        deltas = self.delta_head(decoded)
        last_ball = ball_obs[:, -1:, :2]
        pred = last_ball + torch.cumsum(deltas, dim=1)
        return pred.clamp(0.0, 1.0)


class BallWindowDataset(Dataset):
    """Build training windows for ball prediction from track CSV.

    Expected CSV columns: frame, agent_id, agent_type, x, y.
    The ball is identified by agent_type == 0 or agent_id == 0.
    """

    def __init__(
        self,
        tracks_csv: str,
        obs_steps: int = 20,
        pred_steps: int = 40,
        stride: int = 2,
        max_players: int = 22,
        min_ball_visible_ratio: float = 0.8,
        pitch: PitchSize = PitchSize(),
    ) -> None:
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.total_steps = obs_steps + pred_steps
        self.max_players = max_players
        self.pitch = pitch

        df = pd.read_csv(tracks_csv)
        required = {"frame", "agent_id", "agent_type", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")

        frames = sorted(df["frame"].unique().tolist())
        frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
        ball_df = df[(df["agent_type"] == 0) | (df["agent_id"] == 0)]
        player_df = df[(df["agent_type"] != 0) & (df["agent_id"] != 0)]
        player_ids = (
            player_df.groupby("agent_id")["frame"].nunique().sort_values(ascending=False).head(max_players).index.tolist()
        )
        player_to_idx = {agent_id: idx for idx, agent_id in enumerate(player_ids)}

        self.frames = frames
        self.player_ids = player_ids
        self.players = np.full((len(frames), max_players, 3), np.nan, dtype=np.float32)
        self.player_visible = np.zeros((len(frames), max_players), dtype=np.float32)
        self.ball = np.full((len(frames), 2), np.nan, dtype=np.float32)
        self.ball_visible = np.zeros(len(frames), dtype=np.float32)

        for row in player_df.itertuples(index=False):
            if row.agent_id not in player_to_idx:
                continue
            t = frame_to_idx[row.frame]
            a = player_to_idx[row.agent_id]
            self.players[t, a] = (float(row.x), float(row.y), float(row.agent_type))
            self.player_visible[t, a] = 1.0

        for frame, group in ball_df.groupby("frame"):
            t = frame_to_idx[frame]
            best = group.sort_values("agent_id").iloc[0]
            self.ball[t] = (float(best["x"]), float(best["y"]))
            self.ball_visible[t] = 1.0

        self.windows: list[int] = []
        for start in range(0, max(0, len(frames) - self.total_steps + 1), stride):
            ball_mask = self.ball_visible[start : start + self.total_steps]
            if ball_mask.mean() >= min_ball_visible_ratio and ball_mask[:obs_steps].sum() > 0:
                self.windows.append(start)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.windows[idx]
        end = start + self.total_steps
        players = self.players[start:end].copy()
        player_visible = self.player_visible[start:end].copy()
        ball = self.ball[start:end].copy()
        ball_visible = self.ball_visible[start:end].copy()

        players_xy = _interpolate_array(players[..., :2])
        player_type = np.nan_to_num(players[..., 2:3], nan=1.0) / 2.0
        player_vel = _velocity(players_xy)
        player_features = np.concatenate(
            [
                _normalize_xy(players_xy, self.pitch),
                player_vel / np.array([self.pitch.width, self.pitch.height], dtype=np.float32),
                player_type,
                player_visible[..., None],
            ],
            axis=-1,
        ).astype(np.float32)

        ball_xy = _interpolate_array(ball[:, None, :])[:, 0]
        ball_norm = _normalize_xy(ball_xy, self.pitch)
        ball_vel = _velocity(ball_xy[:, None, :])[:, 0] / np.array([self.pitch.width, self.pitch.height], dtype=np.float32)
        ball_features = np.concatenate([ball_norm, ball_vel, ball_visible[:, None]], axis=-1).astype(np.float32)

        obs_players = player_features[: self.obs_steps]
        obs_ball = ball_features[: self.obs_steps]
        target = ball_norm[self.obs_steps :]
        target_mask = ball_visible[self.obs_steps :].astype(np.float32)
        return torch.from_numpy(obs_players), torch.from_numpy(obs_ball), torch.from_numpy(target), torch.from_numpy(target_mask)


def save_ball_predictions(
    out_csv: str,
    pred_norm: np.ndarray,
    start_frame: int = 0,
    pitch: PitchSize = PitchSize(),
) -> None:
    xy = pred_norm * np.array([pitch.width, pitch.height], dtype=np.float32)
    rows = [
        {"future_step": step + 1, "frame": start_frame + step + 1, "agent_id": 0, "agent_type": 0, "x": float(x), "y": float(y)}
        for step, (x, y) in enumerate(xy)
    ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def build_latest_ball_window(
    tracks_csv: str,
    obs_steps: int,
    max_players: int = 22,
    pitch: PitchSize = PitchSize(),
) -> tuple[torch.Tensor, torch.Tensor, int]:
    dataset = BallWindowDataset(
        tracks_csv,
        obs_steps=obs_steps,
        pred_steps=1,
        stride=1,
        max_players=max_players,
        min_ball_visible_ratio=0.0,
        pitch=pitch,
    )
    if not dataset.frames or len(dataset.frames) < obs_steps:
        raise ValueError(f"Need at least {obs_steps} frames with tracks.")
    start = len(dataset.frames) - obs_steps - 1
    if start < 0:
        start = 0
    dataset.windows = [start]
    players, ball_obs, _target, _mask = dataset[0]
    return players.unsqueeze(0), ball_obs.unsqueeze(0), int(dataset.frames[start + obs_steps - 1])


def masked_ball_ade(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(pred - target, dim=-1)
    return (dist * mask).sum() / mask.sum().clamp_min(1.0)


def _normalize_xy(xy: np.ndarray, pitch: PitchSize) -> np.ndarray:
    return xy.astype(np.float32) / np.array([pitch.width, pitch.height], dtype=np.float32)


def _velocity(xy: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy, dtype=np.float32)
    vel[1:] = xy[1:] - xy[:-1]
    return vel


def _interpolate_array(values: np.ndarray) -> np.ndarray:
    out = values.copy()
    flat = out.reshape(out.shape[0], -1)
    for col in range(flat.shape[1]):
        s = pd.Series(flat[:, col])
        flat[:, col] = s.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)
    return flat.reshape(out.shape).astype(np.float32)
