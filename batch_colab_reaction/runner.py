"""DMF/UMA execution functions."""

from pathlib import Path
import numpy as np
from ase.io import read, write

from .gaussian import convert_traj_to_gaussian_log


def run_dmf(reactant_file, product_file, predictor, output_dir, charge=0, mult=1,
            nmove=5, update_teval=True, convergence="middle"):
    """Run DMF for one reaction and write all result artifacts."""
    from dmf import DirectMaxFlux, interpolate_fbenm
    from fairchem.core import FAIRChemCalculator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ref_images = [read(reactant_file), read(product_file)]
    initial = interpolate_fbenm(ref_images, correlated=True)
    write(output_dir / "DMF_init.xyz", initial.images)
    write(output_dir / "DMF_init.traj", initial.images)
    mxflx = DirectMaxFlux(ref_images, coefs=initial.coefs.copy(), nmove=nmove,
                          update_teval=update_teval)
    for image in mxflx.images:
        image.info["charge"] = charge
        image.info["spin"] = mult
        image.calc = FAIRChemCalculator(predictor, task_name="omol")
    mxflx.add_ipopt_options({"output_file": str(output_dir / "DMF_ipopt.out")})
    mxflx.solve(tol=convergence)
    final_traj = output_dir / "DMF_final.traj"
    write(output_dir / "DMF_final.xyz", mxflx.images)
    write(final_traj, mxflx.images)
    convert_traj_to_gaussian_log(final_traj, output_dir / "DMF_final_gv.log")
    if hasattr(mxflx.history, "images_tmax"):
        write(output_dir / "DMF_tmax.traj", mxflx.history.images_tmax)
    energies = np.array([image.get_potential_energy() for image in mxflx.images])
    ts_index = int(np.argmax(energies))
    ts_image = mxflx.images[ts_index]
    relative_energy = energies[ts_index] - energies[0]
    write(output_dir / "TS_candidate.xyz", ts_image)
    write(output_dir / "TS_candidate.traj", ts_image)
    np.savetxt(output_dir / "energies.txt", np.column_stack([np.arange(len(energies)), energies, energies - energies[0]]),
               header="image energy_eV relative_energy_eV")
    return {"reactant_file": str(reactant_file), "product_file": str(product_file),
            "output_dir": str(output_dir), "images": mxflx.images, "energies": energies,
            "ts_index": ts_index, "ts_image": ts_image, "ts_energy": energies[ts_index],
            "relative_energy": relative_energy, "dmf": mxflx}


def run_all_reactions(reaction_pairs, predictor, output_dir="results", **kwargs):
    """Run every reaction pair, retaining failures in the returned mapping."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for pair in reaction_pairs:
        try:
            results[pair["name"]] = run_dmf(
                pair["reactant"], pair["product"], predictor,
                output_dir / pair["name"], **kwargs
            )
        except Exception as error:
            results[pair["name"]] = {"error": error}
    return results
