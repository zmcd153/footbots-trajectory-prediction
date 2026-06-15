from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def estimate_possession(
    tracks_csv: str,
    out_csv: str,
    distance_threshold: float = 3.0,
    velocity_weight: float = 0.35,
    max_frames: int = 0,
) -> None:
    df = pd.read_csv(tracks_csv)
    required = {"frame", "agent_id", "agent_type", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")

    df = df.sort_values(["frame", "agent_id"]).copy()
    if max_frames > 0:
        df = df[df["frame"] <= int(df["frame"].min()) + max_frames - 1]

    ball = df[(df["agent_type"] == 0) | (df["agent_id"] == 0)].copy()
    players = df[(df["agent_type"] != 0) & (df["agent_id"] != 0)].copy()
    rows: list[dict[str, object]] = []

    if ball.empty:
        for frame in sorted(df["frame"].unique().tolist()):
            rows.append(
                {
                    "frame": int(frame),
                    "ball_x": np.nan,
                    "ball_y": np.nan,
                    "possessor_id": -1,
                    "distance": np.nan,
                    "score": 0.0,
                    "status": "no_ball_detection",
                }
            )
        _save(rows, out_csv)
        return

    ball = ball.sort_values("frame").drop_duplicates("frame", keep="last")
    ball[["ball_vx", "ball_vy"]] = ball[["x", "y"]].diff().fillna(0.0)
    player_vel = _add_velocity(players)
    players = players.merge(player_vel, on=["frame", "agent_id"], how="left")
    players[["vx", "vy"]] = players[["vx", "vy"]].fillna(0.0)

    last_possessor = -1
    for ball_row in ball.itertuples(index=False):
        frame = int(ball_row.frame)
        frame_players = players[players["frame"] == frame]
        if frame_players.empty:
            rows.append(_unknown_row(frame, float(ball_row.x), float(ball_row.y), "no_players"))
            continue

        candidates = []
        ball_xy = np.array([float(ball_row.x), float(ball_row.y)], dtype=np.float32)
        ball_v = np.array([float(ball_row.ball_vx), float(ball_row.ball_vy)], dtype=np.float32)
        for row in frame_players.itertuples(index=False):
            player_xy = np.array([float(row.x), float(row.y)], dtype=np.float32)
            player_v = np.array([float(row.vx), float(row.vy)], dtype=np.float32)
            dist = float(np.linalg.norm(player_xy - ball_xy))
            velocity_match = float(np.linalg.norm(player_v - ball_v))
            continuity_bonus = 0.85 if int(row.agent_id) == last_possessor else 1.0
            cost = (dist + velocity_weight * velocity_match) * continuity_bonus
            candidates.append((cost, dist, int(row.agent_id)))

        candidates.sort(key=lambda x: x[0])
        cost, dist, agent_id = candidates[0]
        score = float(np.exp(-cost / max(distance_threshold, 1e-6)))
        if dist <= distance_threshold:
            last_possessor = agent_id
            status = "controlled"
            possessor_id = agent_id
        else:
            status = "loose"
            possessor_id = -1

        rows.append(
            {
                "frame": frame,
                "ball_x": float(ball_row.x),
                "ball_y": float(ball_row.y),
                "possessor_id": possessor_id,
                "nearest_player_id": agent_id,
                "distance": dist,
                "score": score,
                "status": status,
            }
        )

    _save(rows, out_csv)


def _add_velocity(players: pd.DataFrame) -> pd.DataFrame:
    out = players[["frame", "agent_id", "x", "y"]].copy()
    out = out.sort_values(["agent_id", "frame"])
    out[["prev_frame", "prev_x", "prev_y"]] = out.groupby("agent_id")[["frame", "x", "y"]].shift(1)
    dt = (out["frame"] - out["prev_frame"]).replace(0, np.nan).fillna(1.0)
    out["vx"] = (out["x"] - out["prev_x"]).fillna(0.0) / dt
    out["vy"] = (out["y"] - out["prev_y"]).fillna(0.0) / dt
    return out[["frame", "agent_id", "vx", "vy"]]


def _unknown_row(frame: int, ball_x: float, ball_y: float, status: str) -> dict[str, object]:
    return {
        "frame": frame,
        "ball_x": ball_x,
        "ball_y": ball_y,
        "possessor_id": -1,
        "nearest_player_id": -1,
        "distance": np.nan,
        "score": 0.0,
        "status": status,
    }


def _save(rows: list[dict[str, object]], out_csv: str) -> None:
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    controlled = sum(1 for row in rows if row.get("status") == "controlled")
    print(f"saved possession estimates: {out_csv}")
    print(f"frames: {len(rows)} | controlled: {controlled}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ball possession from ball and player tracks.")
    parser.add_argument("--tracks", default="runs/video_raw_tracks.csv")
    parser.add_argument("--out", default="runs/possession.csv")
    parser.add_argument("--distance-threshold", type=float, default=3.0)
    parser.add_argument("--velocity-weight", type=float, default=0.35)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()
    estimate_possession(
        args.tracks,
        args.out,
        distance_threshold=args.distance_threshold,
        velocity_weight=args.velocity_weight,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
