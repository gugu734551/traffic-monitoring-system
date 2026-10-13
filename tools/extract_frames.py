from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, out_dir: Path, indices: list[int]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    saved: list[Path] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = out_dir / f"frame_{idx:06d}.jpg"
        # Windows 下 OpenCV 的 imwrite 对中文路径偶发失败，这里用 imencode + 写文件更稳
        ok_encode, buf = cv2.imencode(".jpg", frame)
        if not ok_encode:
            raise RuntimeError(f"编码 JPG 失败: frame_idx={idx}")
        try:
            out_path.write_bytes(buf.tobytes())
        except Exception as e:
            raise RuntimeError(f"写入失败: {out_path} ({e})") from e
        saved.append(out_path)

    cap.release()
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="视频路径")
    parser.add_argument("--out", type=str, required=True, help="输出目录")
    parser.add_argument("--frames", type=str, default="", help="逗号分隔的帧号列表，例如: 0,150,300")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if args.frames.strip():
        indices = [int(x) for x in args.frames.split(",") if x.strip()]
    else:
        if frame_count > 0:
            indices = [0, frame_count // 2, max(frame_count - 1, 0)]
        else:
            indices = [0, 150, 300]

    saved = extract_frames(video_path=video_path, out_dir=out_dir, indices=indices)
    print(f"frame_count={frame_count} fps={fps}")
    for p in saved:
        print(str(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

