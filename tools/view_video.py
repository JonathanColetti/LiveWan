"""Render a video as a contact sheet so it can actually be viewed frame-by-frame.

Single PNGs hide temporal failure (flicker, drift, progressive collapse). This
lays every sampled frame out in a labelled grid, optionally marking where
generated frames begin, so the whole clip is visible in one image.

Usage:
  python tools/view_video.py IN.mp4 -o sheet.png [--cols 6] [--max-frames 36]
                             [--mark-from 17] [--label "tag"] [--height 200]
  python tools/view_video.py A.mp4 B.mp4 -o cmp.png --rows-per-video
"""
import argparse, os, sys
import cv2
import numpy as np


def read_frames(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit(f'no frames read from {path}')
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        return [frames[i] for i in idx], list(idx)
    return frames, list(range(len(frames)))


def label(img, text, colour=(255, 255, 255)):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(img, text, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1,
                cv2.LINE_AA)
    return img


def build_sheet(videos, cols, max_frames, mark_from, tile_h, rows_per_video):
    panels = []
    for path in videos:
        frames, idx = read_frames(path, max_frames)
        h, w = frames[0].shape[:2]
        scale = tile_h / h
        tw = max(1, int(round(w * scale)))
        tiles = []
        for f, i in zip(frames, idx):
            t = cv2.resize(f, (tw, tile_h), interpolation=cv2.INTER_AREA)
            gen = mark_from is not None and i >= mark_from
            t = label(t, f'{i}' + (' GEN' if gen else ''),
                      (0, 200, 255) if gen else (255, 255, 255))
            if gen:
                cv2.rectangle(t, (0, 0), (tw - 1, tile_h - 1), (0, 165, 255), 2)
            tiles.append(t)
        ncol = len(tiles) if rows_per_video else cols
        rows = []
        for r in range(0, len(tiles), ncol):
            row = tiles[r:r + ncol]
            while len(row) < ncol:
                row.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(row))
        panel = np.vstack(rows)
        title = np.zeros((22, panel.shape[1], 3), np.uint8)
        cv2.putText(title, f'{os.path.basename(os.path.dirname(path))}/'
                    f'{os.path.basename(path)}  ({len(frames)} shown)',
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 120), 1,
                    cv2.LINE_AA)
        panels.append(np.vstack([title, panel]))
    width = max(p.shape[1] for p in panels)
    padded = [np.hstack([p, np.zeros((p.shape[0], width - p.shape[1], 3), np.uint8)])
              if p.shape[1] < width else p for p in panels]
    return np.vstack(padded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='+')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--max-frames', type=int, default=36)
    ap.add_argument('--mark-from', type=int, default=None,
                    help='frame index at which generated frames begin')
    ap.add_argument('--height', type=int, default=200, help='tile height px')
    ap.add_argument('--rows-per-video', action='store_true',
                    help='one row per video (for side-by-side comparison)')
    a = ap.parse_args()
    sheet = build_sheet(a.videos, a.cols, a.max_frames, a.mark_from, a.height,
                        a.rows_per_video)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cv2.imwrite(a.out, sheet)
    print(f'wrote {a.out}  {sheet.shape[1]}x{sheet.shape[0]}')


if __name__ == '__main__':
    main()
