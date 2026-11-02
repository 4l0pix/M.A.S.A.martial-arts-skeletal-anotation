#CLI entry points
#the same analysis pipeline powers the UI

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from masa.pipeline import RunConfig, analyze_video
from masa.pose import OFFICIAL_MODELS
from masa.video import inspect_video


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="masa", description="Inspect video and export pose tracks"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Print decoded video metadata as JSON")
    inspect_parser.add_argument("video", type=Path)
    analyze_parser = commands.add_parser(
        "analyze", help="Export tracked poses and an H.264 overlay"
    )
    analyze_parser.add_argument("video", type=Path)
    analyze_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New output directory! Existing directories are never overwritten",
    )
    analyze_parser.add_argument(
        "--rights-note",
        required=True,
        help="Document source permission or license and permitted use",
    )
    analyze_parser.add_argument("--source-note", default="")
    analyze_parser.add_argument("--model", choices=OFFICIAL_MODELS, default=OFFICIAL_MODELS[0])
    analyze_parser.add_argument("--tracker", choices=["botsort.yaml", "bytetrack.yaml"], default="botsort.yaml")
    analyze_parser.add_argument("--device", choices=["cpu", "mps", "cuda:0"], default="cpu")
    analyze_parser.add_argument("--start", type=float, default=0.0)
    analyze_parser.add_argument("--end", type=float)
    analyze_parser.add_argument("--stride", type=int, default=1)
    analyze_parser.add_argument("--confidence", type=float, default=0.25)
    analyze_parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    analyze_parser.add_argument("--people", choices=[1, 2], type=int, default=2)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(asdict(inspect_video(args.video)), indent=2))
        else:
            config = RunConfig(
                model=args.model,
                tracker=args.tracker,
                device=args.device,
                start_s=args.start,
                end_s=args.end,
                stride=args.stride,
                confidence=args.confidence,
                keypoint_confidence=args.keypoint_confidence,
                expected_people=args.people,
                rights_note=args.rights_note,
                source_note=args.source_note,
            )
            result = analyze_video(args.video, args.output, config)
            print(
                json.dumps(
                    {
                        "output_dir": str(result.output_dir),
                        "manifest": str(result.manifest_path),
                        "processed_frames": result.frame_count,
                    },
                    indent=2,
                )
            )
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        print(f"masa: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
