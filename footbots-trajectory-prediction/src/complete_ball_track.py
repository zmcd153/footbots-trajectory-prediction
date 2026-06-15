from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def complete_ball_track(
    tracks_csv: str,
    out_csv: str,
    merge_out: str = "",
    possession_csv: str = "",
    max_interp_gap: int = 30,
    proxy_distance: float = 2.0,
    damping: float = 0.92,
    field_width: float = 105.0,
    field_height: float = 68.0,
) -> None:
    tracks = pd.read_csv(tracks_csv)
    required = {"frame", "agent_id", "agent_type", "x", "y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")

    tracks = tracks.sort_values(["frame", "agent_id"]).copy()
    frames = np.arange(int(tracks["frame"].min()), int(tracks["frame"].max()) + 1)
    ball_detections = _extract_ball_detections(tracks)
    possession = _load_possession(possession_csv)
    players_by_frame = {
        int(frame): group.copy()
        for frame, group in tracks[(tracks["agent_type"] != 0) & (tracks["agent_id"] != 0)].groupby("frame")
    }

    completed = pd.DataFrame({"frame": frames})
    completed["x"] = np.nan
    completed["y"] = np.nan
    completed["ball_source"] = "missing"
    completed["ball_confidence"] = 0.0

    if not ball_detections.empty:
        frame_to_idx = {int(frame): idx for idx, frame in enumerate(frames)}
        for row in ball_detections.itertuples(index=False):
            idx = frame_to_idx.get(int(row.frame))
            if idx is None:
                continue
            completed.loc[idx, ["x", "y"]] = [float(row.x), float(row.y)]
            completed.loc[idx, "ball_source"] = "detected"
            completed.loc[idx, "ball_confidence"] = float(getattr(row, "score", 1.0))
        _interpolate_short_gaps(completed, max_interp_gap)

    _predict_missing(
        completed,
        players_by_frame,
        possession,
        proxy_distance=proxy_distance,
        damping=damping,
        field_width=field_width,
        field_height=field_height,
    )

    completed["agent_id"] = 0
    completed["agent_type"] = 0
    completed = completed[["frame", "agent_id", "agent_type", "x", "y", "ball_source", "ball_confidence"]]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    completed.to_csv(out_csv, index=False)
    print(f"saved completed ball track: {out_csv}")
    print(completed["ball_source"].value_counts().to_string())

    if merge_out:
        merged = _merge_ball_into_tracks(tracks, completed)
        Path(merge_out).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(merge_out, index=False)
        print(f"saved merged tracks: {merge_out}")


def _extract_ball_detections(tracks: pd.DataFrame) -> pd.DataFrame:
    ball = tracks[(tracks["agent_type"] == 0) | (tracks["agent_id"] == 0)].copy()
    if ball.empty:
        return ball
    if "score" not in ball.columns:
        ball["score"] = 1.0
    ball["score"] = pd.to_numeric(ball["score"], errors="coerce").fillna(1.0)
    return ball.sort_values(["frame", "score"]).drop_duplicates("frame", keep="last")[["frame", "x", "y", "score"]]


def _interpolate_short_gaps(completed: pd.DataFrame, max_interp_gap: int) -> None:
    detected_idx = completed.index[completed["ball_source"] == "detected"].tolist()
    if len(detected_idx) < 2:
        return
    for left, right in zip(detected_idx[:-1], detected_idx[1:]):
        gap = right - left
        if gap <= 1 or gap > max_interp_gap:
            continue
        x0, y0 = float(completed.loc[left, "x"]), float(completed.loc[left, "y"])
        x1, y1 = float(completed.loc[right, "x"]), float(completed.loc[right, "y"])
        for idx in range(left + 1, right):
            alpha = (idx - left) / gap
            completed.loc[idx, "x"] = x0 * (1.0 - alpha) + x1 * alpha
            completed.loc[idx, "y"] = y0 * (1.0 - alpha) + y1 * alpha
            completed.loc[idx, "ball_source"] = "interpolated"
            completed.loc[idx, "ball_confidence"] = 0.65


def _predict_missing(
    completed: pd.DataFrame,
    players_by_frame: dict[int, pd.DataFrame],
    possession: dict[int, int],
    proxy_distance: float,
    damping: float,
    field_width: float,
    field_height: float,
) -> None:
    prev_xy: np.ndarray | None = None
    velocity = np.zeros(2, dtype=np.float32)
    last_known_idx: int | None = None

    for idx, row in completed.iterrows():
        frame = int(row["frame"])
        if pd.notna(row["x"]) and pd.notna(row["y"]):
            xy = np.array([float(row["x"]), float(row["y"])], dtype=np.float32)
            if prev_xy is not None and last_known_idx is not None:
                dt = max(1, idx - last_known_idx)
                velocity = (xy - prev_xy) / dt
            prev_xy = xy
            last_known_idx = idx
            continue

        proxy_xy = _possessor_proxy(frame, players_by_frame, possession, proxy_distance)
        if proxy_xy is not None:
            xy = proxy_xy
            source = "possessor_proxy"
            confidence = 0.45
            if prev_xy is not None:
                velocity = 0.5 * velocity + 0.5 * (xy - prev_xy)
        elif prev_xy is not None:
            velocity = velocity * damping
            xy = prev_xy + velocity
            source = "predicted"
            confidence = 0.25
        else:
            fallback = _nearest_player_centroid(frame, players_by_frame)
            xy = fallback if fallback is not None else np.array([field_width / 2.0, field_height / 2.0], dtype=np.float32)
            source = "predicted"
            confidence = 0.15

        xy[0] = float(np.clip(xy[0], 0.0, field_width))
        xy[1] = float(np.clip(xy[1], 0.0, field_height))
        completed.loc[idx, "x"] = float(xy[0])
        completed.loc[idx, "y"] = float(xy[1])
        completed.loc[idx, "ball_source"] = source
        completed.loc[idx, "ball_confidence"] = confidence
        prev_xy = xy.astype(np.float32)
        last_known_idx = idx


def _possessor_proxy(
    frame: int,
    players_by_frame: dict[int, pd.DataFrame],
    possession: dict[int, int],
    proxy_distance: float,
) -> np.ndarray | None:
    possessor_id = possession.get(frame, -1)
    if possessor_id < 0:
        return None
    players = players_by_frame.get(frame)
    if players is None or players.empty:
        return None
    player = players[players["agent_id"] == possessor_id]
    if player.empty:
        return None
    row = player.iloc[-1]
    current = np.array([float(row["x"]), float(row["y"])], dtype=np.float32)
    previous = _previous_player_xy(possessor_id, frame, players_by_frame)
    if previous is None:
        return current
    direction = current - previous
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return current
    return current + direction / norm * proxy_distance


def _previous_player_xy(agent_id: int, frame: int, players_by_frame: dict[int, pd.DataFrame]) -> np.ndarray | None:
    for prev_frame in range(frame - 1, frame - 20, -1):
        players = players_by_frame.get(prev_frame)
        if players is None:
            continue
        player = players[players["agent_id"] == agent_id]
        if not player.empty:
            row = player.iloc[-1]
            return np.array([float(row["x"]), float(row["y"])], dtype=np.float32)
    return None


def _nearest_player_centroid(frame: int, players_by_frame: dict[int, pd.DataFrame]) -> np.ndarray | None:
    players = players_by_frame.get(frame)
    if players is None or players.empty:
        return None
    return players[["x", "y"]].mean().to_numpy(dtype=np.float32)


def _load_possession(path: str) -> dict[int, int]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if not {"frame", "possessor_id"}.issubset(df.columns):
        return {}
    return {int(row.frame): int(row.possessor_id) for row in df.itertuples(index=False)}


def _merge_ball_into_tracks(tracks: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    non_ball = tracks[(tracks["agent_type"] != 0) & (tracks["agent_id"] != 0)].copy()
    for column in non_ball.columns:
        if column not in completed.columns:
            completed[column] = np.nan
    for column in completed.columns:
        if column not in non_ball.columns:
            non_ball[column] = np.nan
    merged = pd.concat([non_ball[completed.columns], completed], ignore_index=True)
    return merged.sort_values(["frame", "agent_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete sparse ball detections into a continuous ball track.")
    parser.add_argument("--tracks", default="runs/video_raw_tracks.csv")
    parser.add_argument("--out", default="runs/ball_completed_tracks.csv")
    parser.add_argument("--merge-out", default="runs/video_tracks_with_completed_ball.csv")
    parser.add_argument("--possession", default="", help="Optional possession CSV. Used to place missing ball near the possessor.")
    parser.add_argument("--max-interp-gap", type=int, default=30)
    parser.add_argument("--proxy-distance", type=float, default=2.0)
    parser.add_argument("--damping", type=float, default=0.92)
    parser.add_argument("--field-width", type=float, default=105.0)
    parser.add_argument("--field-height", type=float, default=68.0)
    args = parser.parse_args()
    complete_ball_track(
        args.tracks,
        args.out,
        merge_out=args.merge_out,
        possession_csv=args.possession,
        max_interp_gap=args.max_interp_gap,
        proxy_distance=args.proxy_distance,
        damping=args.damping,
        field_width=args.field_width,
        field_height=args.field_height,
    )


if __name__ == "__main__":
    main()
