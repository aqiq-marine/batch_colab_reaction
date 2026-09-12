"""Reaction input-file discovery and validation."""

from pathlib import Path
import re
from typing import TypedDict


class ReactionPair(TypedDict):
    name: str
    reactant: str
    product: str
    format: str


REACTION_PATTERN = re.compile(
    r"^(.+)_(reactant|product)\.(xyz|pdb)$", re.IGNORECASE
)
SUPPORTED_FORMATS = {"xyz", "pdb"}


def validate_reaction_pairs(pairs: list[ReactionPair]) -> list[ReactionPair]:
    """Validate reaction pairs and return them unchanged."""
    for pair in pairs:
        if pair["format"].lower() not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {pair['format']}")
        if Path(pair["reactant"]).suffix.lower() != Path(pair["product"]).suffix.lower():
            raise ValueError(
                f"Format mismatch for '{pair['name']}': "
                f"{Path(pair['reactant']).name} / {Path(pair['product']).name}"
            )
    return pairs


def scan_reaction_files(input_dir: str | Path = ".") -> list[ReactionPair]:
    """Find ``name_reactant.(xyz|pdb)`` and matching product files."""
    input_dir = Path(input_dir)
    reactions: dict[str, dict[str, Path]] = {}

    for path in input_dir.iterdir():
        if not path.is_file():
            continue
        match = REACTION_PATTERN.match(path.name)
        if match is None:
            continue

        name, role, _extension = match.groups()
        role = role.lower()
        files = reactions.setdefault(name, {})
        if role in files:
            raise ValueError(
                f"Duplicate {role} file for reaction '{name}': "
                f"{files[role]} and {path.name}"
            )
        files[role] = path

    pairs: list[ReactionPair] = []
    for name, files in sorted(reactions.items()):
        if "reactant" not in files:
            raise ValueError(f"Product found without reactant: {name}")
        if "product" not in files:
            raise ValueError(f"Reactant found without product: {name}")
        reactant, product = files["reactant"], files["product"]
        if reactant.suffix.lower() != product.suffix.lower():
            raise ValueError(
                f"Format mismatch for '{name}': {reactant.name} / {product.name}"
            )
        pairs.append({
            "name": name,
            "reactant": str(reactant),
            "product": str(product),
            "format": reactant.suffix.lower().removeprefix("."),
        })
    return pairs
