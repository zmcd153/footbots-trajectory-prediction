from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .ball_trajectory import BallTrajectorySetBiLSTM, BallWindowDataset, masked_ball_ade

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Set Transformer + Hierarchical Bi-LSTM for ball trajectory inference.")
    parser.add_argument("--tracks", required=True, help="Track CSV with player and ball rows.")
    parser.add_argument("--out", default="runs/ball_set_bilstm.pt")
    parser.add_argument("--obs-steps", type=int, default=20)
    parser.add_argument("--pred-steps", type=int, default=40)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=22)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-ball-visible-ratio", type=float, default=0.8)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    dataset = BallWindowDataset(
        args.tracks,
        obs_steps=args.obs_steps,
        pred_steps=args.pred_steps,
        stride=args.stride,
        max_players=args.max_players,
        min_ball_visible_ratio=args.min_ball_visible_ratio,
    )
    if len(dataset) < 2:
        raise RuntimeError("Not enough valid ball windows. Add ball labels or lower --min-ball-visible-ratio.")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BallTrajectorySetBiLSTM(
        obs_steps=args.obs_steps,
        pred_steps=args.pred_steps,
        dim=args.dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for players, ball_obs, target, mask in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            players = players.to(device)
            ball_obs = ball_obs.to(device)
            target = target.to(device)
            mask = mask.to(device)
            pred = model(players, ball_obs)
            loss = masked_ball_ade(pred, target, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for players, ball_obs, target, mask in val_loader:
                players = players.to(device)
                ball_obs = ball_obs.to(device)
                target = target.to(device)
                mask = mask.to(device)
                val_losses.append(masked_ball_ade(model(players, ball_obs), target, mask).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"train_ball_ade_norm={train_loss:.4f} val_ball_ade_norm={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_type": "ball_set_bilstm",
                    "model": model.state_dict(),
                    "obs_steps": args.obs_steps,
                    "pred_steps": args.pred_steps,
                    "max_players": args.max_players,
                    "dim": args.dim,
                    "val_ade_norm": best_val,
                },
                args.out,
            )
            print(f"saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
