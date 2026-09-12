from __future__ import annotations

import argparse
import getpass
import os
import subprocess
from pathlib import Path

from batch_colab_reaction import scan_reaction_files


def authenticate_huggingface(token: str | None = None) -> None:
    token = token or os.environ.get("HF_TOKEN") or getpass.getpass("Hugging Face token: ")
    os.environ["HF_TOKEN"] = token
    subprocess.run(["hf", "auth", "login", "--token", token], check=True)


def load_predictor(model_name: str, device: str):
    from fairchem.core import pretrained_mlip
    return pretrained_mlip.get_predict_unit(model_name, device=device)


def print_detected_reactions(pairs) -> None:
    print("=" * 70)
    print("Detected reactions")
    print("=" * 70)
    for pair in pairs:
        print(f"{pair['name']}: {Path(pair['reactant']).name} <-> {Path(pair['product']).name}")
    print(f"\nTotal: {len(pairs)} reaction(s)")


def print_summary(results) -> None:
    successful = [(name, result) for name, result in results.items() if "error" not in result]
    successful.sort(key=lambda item: item[1]["relative_energy"])
    print("\n" + "=" * 90)
    print("TS Energy Summary (sorted by ΔE‡)")
    print("=" * 90)
    print(f"{'Rank':>6}{'Reaction':<20}{'TS image':>10}{'TS energy (eV)':>20}{'ΔE‡ (eV)':>18}{'ΔE‡ (kcal/mol)':>20}")
    print("-" * 90)
    for rank, (name, result) in enumerate(successful, start=1):
        relative_energy = result["relative_energy"]
        print(f"{rank:>6}{name:<20}{result['ts_index']:>10d}{result['ts_energy']:>20.6f}{relative_energy:>18.6f}{relative_energy * 23.0605:>20.3f}")
    failed = [(name, result) for name, result in results.items() if "error" in result]
    if failed:
        print("\nFailed reactions:")
        for name, result in failed:
            print(f"  ✗ {name}: {type(result['error']).__name__}: {result['error']}")
    print("=" * 90)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--model", default="uma-s-1p2p1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--token", help="Hugging Face token; otherwise HF_TOKEN/getpass is used")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--mult", type=int, default=1)
    parser.add_argument("--nmove", type=int, default=5)
    parser.add_argument("--no-update-teval", action="store_true")
    parser.add_argument("--convergence", default="middle")
    parser.add_argument("--skip-hf-login", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pairs = scan_reaction_files(args.input_dir)
    print_detected_reactions(pairs)
    if not pairs:
        print("No reaction pairs found.")
        return 0
    if not args.skip_hf_login:
        authenticate_huggingface(args.token)
    predictor = load_predictor(args.model, args.device)
    from batch_colab_reaction import run_all_reactions

    results = run_all_reactions(
        pairs, predictor, output_dir=args.output_dir, charge=args.charge,
        mult=args.mult, nmove=args.nmove, update_teval=not args.no_update_teval,
        convergence=args.convergence,
    )
    print_summary(results)
    return 0 if all("error" not in result for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

