from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .ball_trajectory import BallTrajectorySetBiLSTM, build_latest_ball_window, save_ball_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict future ball trajectory from player context and ball history.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", default="runs/ball_predictions.csv")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    obs_steps = int(checkpoint["obs_steps"])
    pred_steps = int(checkpoint["pred_steps"])
    max_players = int(checkpoint.get("max_players", 22))
    dim = int(checkpoint.get("dim", 128))

    players, ball_obs, last_frame = build_latest_ball_window(args.tracks, obs_steps=obs_steps, max_players=max_players)
    model = BallTrajectorySetBiLSTM(obs_steps=obs_steps, pred_steps=pred_steps, dim=dim)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        pred = model(players, ball_obs).squeeze(0).numpy()

    save_ball_predictions(args.out, pred, start_frame=last_frame)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"saved ball predictions: {args.out}")


if __name__ == "__main__":
    main()
