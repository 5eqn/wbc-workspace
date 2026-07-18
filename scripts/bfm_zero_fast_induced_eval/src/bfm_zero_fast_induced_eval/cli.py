from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import OUTPUT_ROOT, EvalConfig
from .onnx_batch import patch_dynamic_batch, verify_dynamic_batches
from .providers import DEFAULT_AMP_TORCHSCRIPT, resolve_provider_artifacts
from .recompute import verify_recomputation
from .runtime import preflight, run_evaluation
from .schema import estimated_raw_gib, raw_bytes_per_accepted_trial


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-provider", choices=("bfm-zero", "legged-lab-amp"), default="bfm-zero")
    common.add_argument("--actor", type=Path)
    common.add_argument("--checkpoint", type=Path)
    common.add_argument("--goals", type=Path)
    common.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    common.add_argument(
        "--allow-custom-model",
        action="store_true",
        help="Allow an actor/checkpoint pair whose hashes differ from the pinned old model.",
    )

    subparsers.add_parser("preflight", parents=[common])
    subparsers.add_parser("patch-onnx", parents=[common])
    run = subparsers.add_parser("run", parents=[common])
    run.add_argument("--run-id", default=None)
    run.add_argument("--rounds", type=int, default=100)
    run.add_argument("--num-envs", type=int, default=128)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--minimum-free-gib", type=float, default=22.0)

    recompute = subparsers.add_parser("verify-recompute", parents=[common])
    recompute.add_argument("round_file", type=Path)
    recompute.add_argument("--samples", type=int, default=8)
    return parser


def main() -> int:
    args = _parser().parse_args()
    actor, checkpoint, goals = resolve_provider_artifacts(
        args.model_provider, args.actor, args.checkpoint, args.goals
    )
    input_dim = 721 if args.model_provider == "bfm-zero" else 570
    if args.command == "verify-recompute":
        from .runtime import reexec_in_isaac

        reexec_in_isaac()
        result = verify_recomputation(args.round_file, checkpoint, args.samples)
    elif args.command == "patch-onnx":
        if args.model_provider == "legged-lab-amp":
            from .runtime import reexec_in_isaac

            reexec_in_isaac()
        patched = patch_dynamic_batch(actor, args.output_root / ".onnx-cache", input_dim, 29)
        torchscript = DEFAULT_AMP_TORCHSCRIPT if args.model_provider == "legged-lab-amp" else None
        result = {
            "patched": str(patched),
            "verification": verify_dynamic_batches(
                actor, patched, input_dim=input_dim, output_dim=29, torchscript=torchscript
            ),
        }
    else:
        config = EvalConfig(
            num_envs=getattr(args, "num_envs", 128),
            rounds=getattr(args, "rounds", 100),
            seed=getattr(args, "seed", 0),
            minimum_free_gib=getattr(args, "minimum_free_gib", 22.0),
        )
        if args.command == "preflight":
            result = preflight(
                args.output_root,
                config,
                actor,
                checkpoint,
                goals,
                allow_custom_model=args.allow_custom_model,
                model_provider=args.model_provider,
            )
            observation_fields = None if args.model_provider == "bfm-zero" else {"provider_input": 570}
            result.update(
                raw_bytes_per_accepted_trial=raw_bytes_per_accepted_trial(observation_fields),
                maximum_raw_gib=estimated_raw_gib(config.num_envs * config.rounds, observation_fields),
            )
        else:
            run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S")
            result = run_evaluation(
                run_id,
                config,
                actor,
                checkpoint,
                goals,
                args.output_root,
                allow_custom_model=args.allow_custom_model,
                model_provider=args.model_provider,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
