from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODE_CONFIG = {
    "straight": {"turn": 0.0, "speed": 1.00, "prob": 0.46},
    "left": {"turn": 0.28, "speed": 0.92, "prob": 0.20},
    "right": {"turn": -0.28, "speed": 0.92, "prob": 0.20},
    "slow": {"turn": 0.0, "speed": 0.45, "prob": 0.14},
}


def predict_multimodal(
    tracks_csv: str,
    out_csv: str,
    pred_steps: int = 40,
    velocity_window: int = 12,
    max_agents: int = 23,
    min_visible: int = 6,
    field_width: float = 105.0,
    field_height: float = 68.0,
    possession_csv: str = "",
    possession_boost: float = 1.25,
    uncertainty_base: float = 0.35,
) -> None:
    df = pd.read_csv(tracks_csv)
    required = {"frame", "agent_id", "agent_type", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")
    df = df.sort_values(["agent_id", "frame"])

    latest_frame = int(df["frame"].max())
    recent = df[df["frame"] >= latest_frame - max(velocity_window * 3, velocity_window)]
    counts = recent.groupby("agent_id")["frame"].nunique().sort_values(ascending=False)
    agent_ids = counts[counts >= min_visible].head(max_agents).index.tolist()
    if not agent_ids:
        raise RuntimeError("No agents have enough recent observations for multimodal prediction.")

    possessor_id = _latest_possessor(possession_csv)
    rows: list[dict[str, object]] = []
    for agent_id in agent_ids:
        g = df[df["agent_id"] == agent_id].sort_values("frame").tail(velocity_window + 1)
        if len(g) < 2:
            continue
        agent_type = int(g["agent_type"].mode().iloc[0]) if "agent_type" in g else 1
        first = g.iloc[0]
        last = g.iloc[-1]
        dt = max(1.0, float(last["frame"] - first["frame"]))
        vx = (float(last["x"]) - float(first["x"])) / dt
        vy = (float(last["y"]) - float(first["y"])) / dt
        speed = float(np.hypot(vx, vy))
        if speed < 0.01:
            vx, vy = 0.0, 0.0

        modes = _agent_modes(int(agent_id), possessor_id, possession_boost)
        for mode_name, cfg in modes.items():
            prob = float(cfg["prob"])
            turn = float(cfg["turn"])
            speed_scale = float(cfg["speed"])
            x = float(last["x"])
            y = float(last["y"])
            mode_vx, mode_vy = _rotate(vx, vy, turn)
            mode_vx *= speed_scale
            mode_vy *= speed_scale
            for step in range(1, pred_steps + 1):
                damping = 0.985 ** (step - 1)
                x = np.clip(x + mode_vx * damping, 0.0, field_width)
                y = np.clip(y + mode_vy * damping, 0.0, field_height)
                uncertainty = uncertainty_base + 0.08 * step + (1.0 - prob) * 1.2 + min(speed, 1.5) * 0.15
                rows.append(
                    {
                        "future_step": step,
                        "mode": mode_name,
                        "mode_probability": prob,
                        "agent_id": int(agent_id),
                        "agent_type": agent_type,
                        "x": float(x),
                        "y": float(y),
                        "uncertainty": float(uncertainty),
                        "possessor_id": int(possessor_id),
                    }
                )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved multimodal predictions: {out_csv}")
    print(f"agents: {len(set(row['agent_id'] for row in rows)) if rows else 0} | modes: {len(MODE_CONFIG)}")


def _latest_possessor(possession_csv: str) -> int:
    if not possession_csv:
        return -1
    path = Path(possession_csv)
    if not path.exists():
        return -1
    df = pd.read_csv(path)
    if "possessor_id" not in df.columns or df.empty:
        return -1
    controlled = df[df["possessor_id"] >= 0]
    if controlled.empty:
        return -1
    return int(controlled.sort_values("frame").iloc[-1]["possessor_id"])


def _agent_modes(agent_id: int, possessor_id: int, boost: float) -> dict[str, dict[str, float]]:
    modes = {name: dict(cfg) for name, cfg in MODE_CONFIG.items()}
    if agent_id == possessor_id:
        modes["straight"]["prob"] *= boost
        modes["left"]["prob"] *= 1.08
        modes["right"]["prob"] *= 1.08
        modes["slow"]["prob"] *= 0.72
    total = sum(float(cfg["prob"]) for cfg in modes.values())
    for cfg in modes.values():
        cfg["prob"] = float(cfg["prob"]) / total
    return modes


def _rotate(vx: float, vy: float, angle: float) -> tuple[float, float]:
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return vx * c - vy * s, vx * s + vy * c


def main() -> None:
    parser = argparse.ArgumentParser(description="Create multiple plausible future trajectories with uncertainty.")
    parser.add_argument("--tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--out", default="runs/video_predictions_multimodal.csv")
    parser.add_argument("--possession", default="", help="Optional possession CSV from src.tactical_features.")
    parser.add_argument("--pred-steps", type=int, default=40)
    parser.add_argument("--velocity-window", type=int, default=12)
    parser.add_argument("--max-agents", type=int, default=23)
    parser.add_argument("--min-visible", type=int, default=6)
    parser.add_argument("--field-width", type=float, default=105.0)
    parser.add_argument("--field-height", type=float, default=68.0)
    parser.add_argument("--uncertainty-base", type=float, default=0.35)
    args = parser.parse_args()
    predict_multimodal(
        args.tracks,
        args.out,
        pred_steps=args.pred_steps,
        velocity_window=args.velocity_window,
        max_agents=args.max_agents,
        min_visible=args.min_visible,
        field_width=args.field_width,
        field_height=args.field_height,
        possession_csv=args.possession,
        uncertainty_base=args.uncertainty_base,
    )


if __name__ == "__main__":
    main()
