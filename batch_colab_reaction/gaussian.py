"""GaussView-compatible output helpers."""

from ase.io import read

EV_TO_HARTREE = 1.0 / 27.2114
HEADER = """\
-----------------------------------------------------------------------
# DirectMaxFlux/UMA
-----------------------------------------------------------------------

IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC

Copyright (c) 2025
Computational Biology Laboratory＠the University of Tokyo
"""
IRC_LINE = "IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC-IRC"


def format_structure(atoms) -> str:
    lines = [IRC_LINE, "                         Input orientation:",
             "---------------------------------------------------------------------",
             "Center     Atomic      Atomic             Coordinates (Angstroms)",
             "Number     Number       Type             X           Y           Z",
             "---------------------------------------------------------------------"]
    for i, atom in enumerate(atoms):
        x, y, z = atom.position
        lines.append(f"{i + 1:5d}\t{atom.number:<2d}\t0\t{x: .10f}\t{y: .10f}\t{z: .10f}")
    lines.append("---------------------------------------------------------------------")
    return "\n".join(lines)


def convert_traj_to_gaussian_log(traj_file, output_log) -> None:
    traj = read(traj_file, ":")
    if not traj:
        raise ValueError(f"No frames found in trajectory: {traj_file}")
    last_energy = 0.0
    with open(output_log, "w", encoding="utf-8") as stream:
        stream.write(HEADER + "\n\n")
        for i, atoms in enumerate(traj):
            try:
                energy = float(atoms.get_potential_energy()) * EV_TO_HARTREE
                last_energy = energy
            except Exception:
                energy = last_energy
            if i:
                point = i - 1
                stream.write(f"{IRC_LINE}\nPt {point} Step number   1 out of a maximum of  1\n")
                stream.write(f"NET REACTION COORDINATE UP TO THIS POINT = {float(point):20.10f}\n\n")
            stream.write(format_structure(atoms) + "\n")
            stream.write(f"SCF Done:  E(scf) =  {energy: .10f}     A.U.\n\n")
        point = len(traj) - 1
        stream.write(f"{IRC_LINE}\nPt {point} Step number   1 out of a maximum of  1\n")
        stream.write(f"NET REACTION COORDINATE UP TO THIS POINT = {float(point):20.10f}\n\n")
        stream.write(f"{IRC_LINE}\nNormal termination of Gaussian\n")
