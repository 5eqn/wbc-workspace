#!/usr/bin/env python3
"""Root-owned CLI for the WBC benchmark."""

from __future__ import annotations

import argparse
from typing import Iterable

from benchmark_common import MOTIONS, POLICIES, ROOT
from benchmark_checks import (
    docker,
    prepare_stock_assets,
    smoke_holomotion,
    smoke_release_gate,
    smoke_sim_bridge,
    smoke_sim_release,
    smoke_sonic_build,
    validate_assets,
    validate_deploy,
    validate_images,
)
from benchmark_report import report
from benchmark_runtime import run_all_motions, run_motion, run_release_validation


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate-assets").set_defaults(func=validate_assets)
    sub.add_parser("prepare-assets").set_defaults(func=prepare_stock_assets)
    sub.add_parser("validate-deploy").set_defaults(func=validate_deploy)
    sub.add_parser("validate-images").set_defaults(func=validate_images)
    sub.add_parser("smoke-release-gate").set_defaults(func=smoke_release_gate)
    sub.add_parser("smoke-sim-bridge").set_defaults(func=smoke_sim_bridge)
    sub.add_parser("smoke-sim-release").set_defaults(func=smoke_sim_release)
    sub.add_parser("smoke-holomotion").set_defaults(func=smoke_holomotion)
    sub.add_parser("smoke-sonic-build").set_defaults(func=smoke_sonic_build)
    p = sub.add_parser("run-release-validation")
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--policy-ready-timeout-s", type=float, default=120.0)
    p.add_argument("--support-height", type=float, default=0.75)
    p.add_argument("--motion", choices=MOTIONS, default=MOTIONS[0])
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.set_defaults(func=run_release_validation)
    p = sub.add_parser("report")
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    p.set_defaults(func=report)
    p = sub.add_parser("run-motion")
    p.add_argument("policy", choices=POLICIES)
    p.add_argument("motion", choices=MOTIONS)
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--startup-margin-s", type=float, default=120.0)
    p.add_argument("--policy-ready-timeout-s", type=float, default=120.0)
    p.add_argument("--holomotion-default-wait-s", type=float, default=5.0)
    p.add_argument("--sonic-post-release-wait-s", type=float, default=2.0)
    p.add_argument("--support-height", type=float, default=0.75)
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.set_defaults(func=run_motion)
    p = sub.add_parser("run-all-motions")
    p.add_argument("--logs", default=str(ROOT / "logs"))
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--startup-margin-s", type=float, default=120.0)
    p.add_argument("--policy-ready-timeout-s", type=float, default=120.0)
    p.add_argument("--holomotion-default-wait-s", type=float, default=5.0)
    p.add_argument("--sonic-post-release-wait-s", type=float, default=2.0)
    p.add_argument("--support-height", type=float, default=0.75)
    p.add_argument("--skip-gpu-preflight", action="store_true")
    p.set_defaults(func=run_all_motions)
    p = sub.add_parser("docker-build")
    p.add_argument("image", choices=["gear-sonic", "holomotion", "unitree_mujoco"])
    p.set_defaults(func=docker)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
