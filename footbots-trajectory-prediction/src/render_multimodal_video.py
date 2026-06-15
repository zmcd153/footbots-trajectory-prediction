from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PALETTE = [
    (40, 220, 255), (80, 180, 255), (120, 255, 120), (255, 180, 80),
    (255, 120, 180), (180, 120, 255), (255, 255, 80), (80, 255, 180),
    (255, 100, 100), (180, 255, 80),
]

MODE_STYLE = {
    "straight": 1.0,
    "left": 0.75,
    "right": 0.75,
    "slow": 0.55,
}


def color_for(agent_id: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(agent_id)) % len(PALETTE)]


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_polyline(frame: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pts) < 2:
        return
    cv2.polylines(frame, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def render(args: argparse.Namespace) -> None:
    tracks = pd.read_csv(args.tracks)
    required = {"frame", "agent_id", "x", "y", "image_x", "image_y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Tracks are missing columns: {sorted(missing)}. Re-run main.py to regenerate tracks.")
    tracks = tracks.sort_values(["frame", "agent_id"])
    by_frame = {int(frame): group for frame, group in tracks.groupby("frame")}

    predictions = pd.read_csv(args.predictions)
    pred_required = {"future_step", "mode", "mode_probability", "agent_id", "x", "y", "uncertainty"}
    pred_missing = pred_required - set(predictions.columns)
    if pred_missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(pred_missing)}")

    possession = _load_possession(args.possession)
    anchors = _latest_anchors(tracks)
    pred_pixels = _predictions_to_pixels(predictions, anchors, args.px_per_meter, args.prediction_scale)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
    last_frame: np.ndarray | None = None
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame.copy()
        rows = by_frame.get(frame_idx)
        possessor_id = possession.get(frame_idx, -1)
        if rows is not None:
            for row in rows.itertuples(index=False):
                agent_id = int(row.agent_id)
                pt = (int(round(float(row.image_x))), int(round(float(row.image_y))))
                trails[agent_id].append(pt)
                color = color_for(agent_id)
                draw_polyline(frame, list(trails[agent_id]), color, 2)
                radius = 8 if agent_id == possessor_id else 5
                cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
                if agent_id == possessor_id:
                    cv2.circle(frame, pt, radius + 5, (255, 255, 255), 2, cv2.LINE_AA)
                    draw_label(frame, f"poss {agent_id}", (pt[0] + 9, pt[1] - 10), (255, 255, 255))
                else:
                    draw_label(frame, str(agent_id), (pt[0] + 7, pt[1] - 7), color)
        draw_label(frame, f"frame {frame_idx}", (18, 32), (255, 255, 255))
        writer.write(frame)
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()

    if last_frame is not None:
        freeze_frames = max(1, int(round(args.prediction_seconds * fps)))
        for i in range(freeze_frames):
            alpha = min(1.0, (i + 1) / max(1, freeze_frames // 2))
            frame = last_frame.copy()
            overlay = frame.copy()
            for (agent_id, mode), items in pred_pixels.items():
                color = _scale_color(color_for(agent_id), MODE_STYLE.get(mode, 0.65))
                visible_count = max(2, int(round(len(items) * alpha)))
                visible = items[:visible_count]
                pts = [item["pt"] for item in visible]
                if args.draw_uncertainty:
                    for item in visible[:: max(1, args.uncertainty_every)]:
                        radius = int(round(item["uncertainty"] * args.px_per_meter * args.uncertainty_scale))
                        if radius > 1:
                            cv2.circle(overlay, item["pt"], radius, color, 1, cv2.LINE_AA)
                draw_polyline(overlay, pts, color, max(1, args.pred_thickness))
                if pts:
                    end = pts[-1]
                    cv2.circle(overlay, end, 3, color, -1, cv2.LINE_AA)
                    if mode == "straight":
                        draw_label(overlay, f"{agent_id} p={visible[-1]['prob']:.2f}", (pts[0][0] + 8, pts[0][1] - 8), color)
            cv2.addWeighted(overlay, 0.86, frame, 0.14, 0, frame)
            draw_label(frame, "multimodal prediction + uncertainty", (18, 32), (255, 255, 255))
            writer.write(frame)

    writer.release()
    print(f"saved video: {out}")
    print(f"source frames read: {frame_idx}/{total_frames if total_frames else '?'}")


def _latest_anchors(tracks: pd.DataFrame) -> dict[int, tuple[float, float, float, float]]:
    anchors = {}
    for agent_id, group in tracks.groupby("agent_id"):
        last = group.sort_values("frame").iloc[-1]
        anchors[int(agent_id)] = (float(last.x), float(last.y), float(last.image_x), float(last.image_y))
    return anchors


def _predictions_to_pixels(
    predictions: pd.DataFrame,
    anchors: dict[int, tuple[float, float, float, float]],
    px_per_meter: float,
    scale: float,
) -> dict[tuple[int, str], list[dict[str, object]]]:
    out: dict[tuple[int, str], list[dict[str, object]]] = defaultdict(list)
    predictions = predictions.sort_values(["agent_id", "mode", "future_step"])
    for (agent_id, mode), group in predictions.groupby(["agent_id", "mode"]):
        agent_id = int(agent_id)
        if agent_id not in anchors:
            continue
        last_x, last_y, anchor_px, anchor_py = anchors[agent_id]
        for row in group.itertuples(index=False):
            dx = (float(row.x) - last_x) * px_per_meter * scale
            dy = (float(row.y) - last_y) * px_per_meter * scale
            out[(agent_id, str(mode))].append(
                {
                    "pt": (int(round(anchor_px + dx)), int(round(anchor_py + dy))),
                    "uncertainty": float(row.uncertainty),
                    "prob": float(row.mode_probability),
                }
            )
    return out


def _load_possession(path: str) -> dict[int, int]:
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if not {"frame", "possessor_id"}.issubset(df.columns):
        return {}
    return {int(row.frame): int(row.possessor_id) for row in df.itertuples(index=False)}


def _scale_color(color: tuple[int, int, int], scale: float) -> tuple[int, int, int]:
    return tuple(int(np.clip(c * scale, 0, 255)) for c in color)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render possession, multimodal predictions, and uncertainty on video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--predictions", default="runs/video_predictions_multimodal.csv")
    parser.add_argument("--possession", default="runs/possession.csv")
    parser.add_argument("--out", default="runs/video_predictions_multimodal_uncertainty.mp4")
    parser.add_argument("--trail", type=int, default=30)
    parser.add_argument("--prediction-seconds", type=float, default=4.0)
    parser.add_argument("--px-per-meter", type=float, default=34.7)
    parser.add_argument("--prediction-scale", type=float, default=1.0)
    parser.add_argument("--uncertainty-scale", type=float, default=0.45)
    parser.add_argument("--uncertainty-every", type=int, default=5)
    parser.add_argument("--draw-uncertainty", action="store_true")
    parser.add_argument("--pred-thickness", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fourcc", default="mp4v")
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()
