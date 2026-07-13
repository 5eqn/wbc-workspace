from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HeraclesConfig
from .data import preprocess_dataset
from .evaluation import summarize_trials
from .inference import benchmark_inference, export_onnx
from .model import HeraclesPlanner
from .serve import serve
from .training import benchmark_training, train

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "thirdparties/BFM-Zero/humanoidverse/data/lafan_29dof.pkl"
DEFAULT_MODEL_XML = (
    ROOT
    / "thirdparties/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/mjcf"
    / "g1_29dof_rev_1_0.xml"
)
DEFAULT_DATA = ROOT / "logs/heracles-planner/data"
DEFAULT_ARTIFACTS = ROOT / "artifacts/heracles-planner"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="35D Heracles planner reproduction")
    sub = parser.add_subparsers(dest="command", required=True)
    preprocess = sub.add_parser("preprocess")
    preprocess.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    preprocess.add_argument("--model-xml", type=Path, default=DEFAULT_MODEL_XML)
    preprocess.add_argument("--output", type=Path, default=DEFAULT_DATA)
    inspect = sub.add_parser("inspect-model")
    inspect.add_argument("--output", type=Path)
    benchmark = sub.add_parser("benchmark-training")
    benchmark.add_argument("--data", type=Path, default=DEFAULT_DATA)
    benchmark.add_argument("--steps", type=int, default=50)
    benchmark.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACTS / "training_duration_gate.json"
    )
    training = sub.add_parser("train")
    training.add_argument("--data", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "checkpoints")
    training.add_argument(
        "--duration-report",
        type=Path,
        default=DEFAULT_ARTIFACTS / "training_duration_gate.json",
    )
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True)
    export.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "heracles.onnx")
    export.add_argument(
        "--parity-output", type=Path, default=DEFAULT_ARTIFACTS / "onnx_parity.json"
    )
    inference = sub.add_parser("benchmark-inference")
    inference.add_argument("--model", type=Path, required=True)
    inference.add_argument("--normalization", type=Path, default=DEFAULT_DATA / "normalization.npz")
    inference.add_argument("--iterations", type=int, default=200)
    inference.add_argument("--output", type=Path, default=DEFAULT_ARTIFACTS / "inference_25hz.json")
    server = sub.add_parser("serve")
    server.add_argument("--model", type=Path, required=True)
    server.add_argument("--normalization", type=Path, default=DEFAULT_DATA / "normalization.npz")
    server.add_argument("--motion", type=Path, required=True)
    server.add_argument("--state-endpoint", default="tcp://127.0.0.1:15558")
    server.add_argument("--pose-endpoint", default="tcp://*:5556")
    server.add_argument("--log", type=Path, required=True)
    server.add_argument("--max-seconds", type=float)
    summarize = sub.add_parser("summarize-evaluation")
    summarize.add_argument("trials", nargs="+", type=Path)
    summarize.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = HeraclesConfig()
    if args.command == "preprocess":
        result = preprocess_dataset(args.source, args.output, args.model_xml, config)
        print(json.dumps({"motions": len(result["motions"]), "output": str(args.output)}, indent=2))
    elif args.command == "inspect-model":
        model = HeraclesPlanner(config)
        result = {"parameter_count": model.parameter_count, "config": config.to_dict()}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    elif args.command == "benchmark-training":
        print(json.dumps(benchmark_training(args.data, args.output, args.steps, config), indent=2))
    elif args.command == "train":
        train(args.data, args.output, args.duration_report, config)
    elif args.command == "export":
        print(json.dumps(export_onnx(args.checkpoint, args.output, args.parity_output), indent=2))
    elif args.command == "benchmark-inference":
        print(
            json.dumps(
                benchmark_inference(args.model, args.normalization, args.output, args.iterations),
                indent=2,
            )
        )
    elif args.command == "serve":
        serve(
            args.model,
            args.normalization,
            args.motion,
            args.state_endpoint,
            args.pose_endpoint,
            args.log,
            args.max_seconds,
        )
    elif args.command == "summarize-evaluation":
        print(json.dumps(summarize_trials(args.trials, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
