"""Utilities for batch transition-state calculations."""

from .files import ReactionPair, scan_reaction_files, validate_reaction_pairs

# Keep file discovery usable without installing the scientific runtime.
def __getattr__(name):
    if name in {"format_structure", "convert_traj_to_gaussian_log"}:
        from .gaussian import convert_traj_to_gaussian_log, format_structure
        return {"format_structure": format_structure,
                "convert_traj_to_gaussian_log": convert_traj_to_gaussian_log}[name]
    if name in {"run_dmf", "run_all_reactions"}:
        from .runner import run_all_reactions, run_dmf
        return {"run_dmf": run_dmf, "run_all_reactions": run_all_reactions}[name]
    raise AttributeError(name)

__all__ = [
    "ReactionPair",
    "scan_reaction_files",
    "validate_reaction_pairs",
    "format_structure",
    "convert_traj_to_gaussian_log",
    "run_dmf",
    "run_all_reactions",
]
