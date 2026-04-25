#!/usr/bin/env python3
"""
Orchestrator script for Resonance Transformer experiments.

This script:
  1. Trains tiny StandardTransformer and ResonanceTransformer baselines
  2. Runs all three experimental harnesses
  3. Summarizes results

Usage:
    python run_experiments.py --epochs 10 --device cuda
"""
import argparse
import os
import subprocess
import sys


def run_command(cmd: list, description: str):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd="/mnt/agents/output/resonance")
    if result.returncode != 0:
        print(f"WARNING: {description} exited with code {result.returncode}")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run all Resonance Transformer experiments")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs for baseline models")
    parser.add_argument("--device", type=str, default="cuda" if sys.platform != "darwin" else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--skip_training", action="store_true", help="Skip training if checkpoints exist")
    args = parser.parse_args()

    base_dir = "/mnt/agents/output/resonance"
    os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "figures"), exist_ok=True)

    # 1. Train baselines
    if not args.skip_training:
        print("\n[Orchestrator] Training baseline models...")
        # We train via the perturbation script with --train_if_missing
        # But to be explicit, we can just let each experiment script train.
        # Instead, let's pre-train both models.
        train_script = os.path.join(base_dir, "experiments", "perturbation_stability.py")
        # Train standard
        run_command([
            sys.executable, train_script,
            "--model_type", "standard",
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--output_json", "results/perturbation_standard.json",
            "--output_plot", "figures/perturbation_standard.png",
        ], "Train StandardTransformer baseline")
        # Train resonance
        run_command([
            sys.executable, train_script,
            "--model_type", "resonance",
            "--device", args.device,
            "--batch_size", str(args.batch_size),
            "--output_json", "results/perturbation_resonance.json",
            "--output_plot", "figures/perturbation_resonance.png",
        ], "Train ResonanceTransformer baseline")

    # 2. Run perturbation stability (already partly done during training, but re-run cleanly)
    print("\n[Orchestrator] Running perturbation stability experiments...")
    run_command([
        sys.executable, os.path.join(base_dir, "experiments", "perturbation_stability.py"),
        "--model_type", "standard",
        "--model_path", "checkpoints/standard.pt",
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--output_json", "results/perturbation_standard.json",
        "--output_plot", "figures/perturbation_standard.png",
    ], "Perturbation Stability - Standard")
    run_command([
        sys.executable, os.path.join(base_dir, "experiments", "perturbation_stability.py"),
        "--model_type", "resonance",
        "--model_path", "checkpoints/resonance.pt",
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--output_json", "results/perturbation_resonance.json",
        "--output_plot", "figures/perturbation_resonance.png",
    ], "Perturbation Stability - Resonance")

    # 3. Run parameter compressibility
    print("\n[Orchestrator] Running parameter compressibility experiment...")
    run_command([
        sys.executable, os.path.join(base_dir, "experiments", "param_compressibility.py"),
        "--standard_path", "checkpoints/standard.pt",
        "--resonance_path", "checkpoints/resonance.pt",
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--output_json", "results/compressibility_results.json",
        "--output_plot", "figures/compressibility.png",
    ], "Parameter Compressibility")

    # 4. Run gestalt compressibility
    print("\n[Orchestrator] Running gestalt compressibility experiment...")
    run_command([
        sys.executable, os.path.join(base_dir, "experiments", "gestalt_compressibility.py"),
        "--model_path", "checkpoints/resonance.pt",
        "--device", args.device,
        "--batch_size", str(args.batch_size),
        "--output_json", "results/gestalt_results.json",
        "--output_plot", "figures/gestalt_analysis.png",
    ], "Gestalt Compressibility")

    print("\n[Orchestrator] All experiments complete!")
    print("Results:")
    print("  - results/perturbation_standard.json")
    print("  - results/perturbation_resonance.json")
    print("  - results/compressibility_results.json")
    print("  - results/gestalt_results.json")
    print("Figures:")
    print("  - figures/perturbation_standard.png")
    print("  - figures/perturbation_resonance.png")
    print("  - figures/compressibility.png")
    print("  - figures/gestalt_analysis.png")


if __name__ == "__main__":
    main()
