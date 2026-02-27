#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch inference runner for PanNuke: run inference sequentially on training runs
that are missing results. Safely skips runs that already have inference output.

Usage examples:
  # Dry run: see what would run without executing
  python run_missing_inference.py --root /path/to/trainings --dry_run

  # Limit to first 5 runs (useful for testing)
  python run_missing_inference.py --root /path/to/trainings --max_runs 5

  # Pass extra args to inference script (e.g. --plots)
  python run_missing_inference.py --root /path/to/trainings --extra_args --plots

  # Full SCC wrapper (loads modules, conda, then runs this)
  ./run_missing_inference_scc.sh
  ./run_missing_inference_scc.sh --dry_run
  ./run_missing_inference_scc.sh --max_runs 3
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def flush_print(*args, **kwargs) -> None:
    """Print and flush immediately for live output in batch jobs."""
    print(*args, **kwargs, flush=True)


def _split_extra_args(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Split argv at --extra_args: (our_args, inference_args)."""
    if "--extra_args" in argv:
        idx = argv.index("--extra_args")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _is_run_dir(path: Path) -> bool:
    """
    Treat a folder as a run dir if it contains any of:
    checkpoints/ OR config.yaml OR hparams.yaml OR any *.pth/*.ckpt somewhere inside.
    """
    if not path.is_dir():
        return False
    if (path / "checkpoints").is_dir():
        return True
    if (path / "config.yaml").is_file():
        return True
    if (path / "hparams.yaml").is_file():
        return True
    for ext in ("*.pth", "*.ckpt"):
        if any(path.rglob(ext)):
            return True
    return False


def _has_any_result(run_dir: Path, result_names: List[str]) -> bool:
    """Check if run_dir contains any of the result filenames."""
    for name in result_names:
        if (run_dir / name).is_file():
            return True
    return False


def _has_checkpoint(run_dir: Path, checkpoint_name: str) -> bool:
    """Check if run_dir has the required checkpoint file."""
    return (run_dir / "checkpoints" / checkpoint_name).is_file()


def _create_symlink_if_needed(run_dir: Path) -> None:
    """
    If inference_results_normal.json exists and inference_results.json does not,
    create a relative symlink inference_results.json -> inference_results_normal.json.
    Warn on failure but do not crash.
    """
    normal_path = run_dir / "inference_results_normal.json"
    legacy_path = run_dir / "inference_results.json"
    if not normal_path.exists() or legacy_path.exists():
        return
    try:
        legacy_path.symlink_to("inference_results_normal.json")
        flush_print(f"  -> Created symlink inference_results.json -> inference_results_normal.json")
    except OSError as e:
        flush_print(f"  WARNING: Could not create symlink: {e}")


def main() -> int:
    our_argv, inference_extra = _split_extra_args(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Run PanNuke inference for training runs missing results (sequential, single GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing run directories (scanned 1 level deep)",
    )
    parser.add_argument(
        "--inference_py",
        type=str,
        required=True,
        help="Path to inference_cellvit_experiment_pannuke.py",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id for inference",
    )
    parser.add_argument(
        "--result_names",
        type=str,
        nargs="+",
        default=["inference_results.json", "inference_results_normal.json"],
        help="Result filenames to treat as 'done'; skip run if any exist",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print what would run, do not execute",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=0,
        help="Max number of runs to execute (0 = no limit)",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="model_best.pth",
        help="Checkpoint filename; skip run dirs that don't have checkpoints/<name>",
    )
    parser.epilog = (
        "Use --extra_args to pass additional args to the inference script. "
        "Example: --extra_args --plots"
    )
    args = parser.parse_args(our_argv)

    root = Path(args.root).resolve()
    inference_py = Path(args.inference_py).resolve()

    if not root.is_dir():
        flush_print(f"ERROR: --root is not a directory: {root}", file=sys.stderr)
        return 1
    if not inference_py.is_file():
        flush_print(f"ERROR: --inference_py not found: {inference_py}", file=sys.stderr)
        return 1

    # Scan 1 level deep
    all_folders = sorted([p for p in root.iterdir() if p.is_dir()])
    run_dirs = [p for p in all_folders if _is_run_dir(p)]
    skipped_not_run = len(all_folders) - len(run_dirs)
    already_done = [p for p in run_dirs if _has_any_result(p, args.result_names)]
    missing_results = [p for p in run_dirs if not _has_any_result(p, args.result_names)]
    skipped_has_results = len(already_done)
    # Skip run dirs that don't have the required checkpoint
    to_run = [p for p in missing_results if _has_checkpoint(p, args.checkpoint_name)]
    skipped_no_checkpoint = len(missing_results) - len(to_run)

    flush_print("=" * 60)
    flush_print("Batch inference summary")
    flush_print("=" * 60)
    flush_print(f"Total folders (1 level under root): {len(all_folders)}")
    flush_print(f"Skipped (not run dirs):             {skipped_not_run}")
    flush_print(f"Skipped (already has results):      {skipped_has_results}")
    flush_print(f"Skipped (no checkpoint):            {skipped_no_checkpoint}")
    flush_print(f"To run:                             {len(to_run)}")
    flush_print("=" * 60)

    if args.max_runs > 0:
        to_run = to_run[: args.max_runs]
        flush_print(f"Limited to first {args.max_runs} runs")
        flush_print("-" * 60)

    if not to_run:
        flush_print("Nothing to run.")
        return 0

    if args.dry_run:
        flush_print("DRY RUN - would run inference on:")
        for p in to_run:
            flush_print(f"  {p}")
        flush_print(f"Total: {len(to_run)} run(s)")
        return 0

    python = sys.executable
    for i, run_dir in enumerate(to_run, 1):
        flush_print(f"[{i}/{len(to_run)}] Running: {run_dir}")
        cmd = [
            python,
            str(inference_py),
            "--run_dir",
            str(run_dir),
            "--gpu",
            str(args.gpu),
            "--checkpoint_name",
            args.checkpoint_name,
        ] + inference_extra
        ret = subprocess.run(cmd)
        if ret.returncode != 0:
            flush_print(
                f"ERROR: Inference failed for {run_dir} (exit code {ret.returncode})",
                file=sys.stderr,
            )
            return ret.returncode
        _create_symlink_if_needed(run_dir)

    flush_print(f"Completed {len(to_run)} run(s) successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
