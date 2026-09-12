"""DMF/UMA execution functions."""

from pathlib import Path
import logging
import time

import numpy as np
from ase.io import read, write

from .gaussian import convert_traj_to_gaussian_log


logger = logging.getLogger(__name__)


def run_dmf(
    reactant_file,
    product_file,
    predictor,
    output_dir,
    charge=0,
    mult=1,
    nmove=5,
    update_teval=True,
    convergence="middle",
):
    """Run DMF for one reaction and write all result artifacts."""

    from dmf import DirectMaxFlux, interpolate_fbenm
    from fairchem.core import FAIRChemCalculator

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()

    logger.info("=" * 70)
    logger.info("Starting DMF calculation")
    logger.info("  Reactant : %s", reactant_file)
    logger.info("  Product  : %s", product_file)
    logger.info("  Output   : %s", output_dir)
    logger.info(
        "  charge=%s, multiplicity=%s, nmove=%s, "
        "update_teval=%s, convergence=%s",
        charge,
        mult,
        nmove,
        update_teval,
        convergence,
    )

    # ------------------------------------------------------------------
    # Read structures
    # ------------------------------------------------------------------
    logger.info("[1/8] Reading reactant/product structures...")
    ref_images = [read(reactant_file), read(product_file)]

    logger.info(
        "      Reactant: %d atoms",
        len(ref_images[0]),
    )
    logger.info(
        "      Product : %d atoms",
        len(ref_images[1]),
    )

    # ------------------------------------------------------------------
    # Initial interpolation
    # ------------------------------------------------------------------
    logger.info("[2/8] Generating initial DMF path...")
    t0 = time.perf_counter()

    initial = interpolate_fbenm(ref_images, correlated=True)

    logger.info(
        "      Generated %d images (%.2f s)",
        len(initial.images),
        time.perf_counter() - t0,
    )

    write(output_dir / "DMF_init.xyz", initial.images)
    write(output_dir / "DMF_init.traj", initial.images)

    logger.info("      Saved DMF_init.xyz / DMF_init.traj")

    # ------------------------------------------------------------------
    # Create DMF object
    # ------------------------------------------------------------------
    logger.info("[3/8] Initializing DirectMaxFlux...")
    mxflx = DirectMaxFlux(
        ref_images,
        coefs=initial.coefs.copy(),
        nmove=nmove,
        update_teval=update_teval,
    )

    logger.info(
        "      DMF initialized with %d images",
        len(mxflx.images),
    )

    # ------------------------------------------------------------------
    # Configure calculator
    # ------------------------------------------------------------------
    logger.info("[4/8] Configuring UMA calculators...")

    for i, image in enumerate(mxflx.images):
        image.info["charge"] = charge
        image.info["spin"] = mult
        image.calc = FAIRChemCalculator(
            predictor,
            task_name="omol",
        )

        logger.debug(
            "      Calculator assigned to image %d",
            i,
        )

    mxflx.add_ipopt_options(
        {
            "output_file": str(output_dir / "DMF_ipopt.out")
        }
    )

    logger.info(
        "      IPOPT output: %s",
        output_dir / "DMF_ipopt.out",
    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    logger.info("[5/8] Starting DMF optimization...")
    logger.info(
        "      This may take some time. "
        "Detailed IPOPT output is written to DMF_ipopt.out"
    )

    solve_start = time.perf_counter()

    try:
        mxflx.solve(tol=convergence)
    except Exception:
        logger.exception(
            "DMF optimization failed after %.2f s",
            time.perf_counter() - solve_start,
        )
        raise

    logger.info(
        "      DMF optimization finished (%.2f s)",
        time.perf_counter() - solve_start,
    )

    # ------------------------------------------------------------------
    # Save final trajectory
    # ------------------------------------------------------------------
    logger.info("[6/8] Saving final DMF structures...")

    final_traj = output_dir / "DMF_final.traj"

    write(output_dir / "DMF_final.xyz", mxflx.images)
    write(final_traj, mxflx.images)

    logger.info(
        "      Saved %d final images",
        len(mxflx.images),
    )

    convert_traj_to_gaussian_log(
        final_traj,
        output_dir / "DMF_final_gv.log",
    )

    logger.info("      Saved DMF_final_gv.log")

    if hasattr(mxflx.history, "images_tmax"):
        write(
            output_dir / "DMF_tmax.traj",
            mxflx.history.images_tmax,
        )
        logger.info("      Saved DMF_tmax.traj")

    # ------------------------------------------------------------------
    # Analyze energies
    # ------------------------------------------------------------------
    logger.info("[7/8] Analyzing image energies...")

    energies = np.array(
        [
            image.get_potential_energy()
            for image in mxflx.images
        ]
    )

    ts_index = int(np.argmax(energies))
    ts_image = mxflx.images[ts_index]

    relative_energy = (
        energies[ts_index] - energies[0]
    )

    logger.info(
        "      Reactant energy : %.6f eV",
        energies[0],
    )
    logger.info(
        "      TS candidate    : image %d",
        ts_index,
    )
    logger.info(
        "      TS energy       : %.6f eV",
        energies[ts_index],
    )
    logger.info(
        "      Relative energy : %.6f eV",
        relative_energy,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    logger.info("[8/8] Saving analysis results...")

    write(
        output_dir / "TS_candidate.xyz",
        ts_image,
    )
    write(
        output_dir / "TS_candidate.traj",
        ts_image,
    )

    np.savetxt(
        output_dir / "energies.txt",
        np.column_stack(
            [
                np.arange(len(energies)),
                energies,
                energies - energies[0],
            ]
        ),
        header="image energy_eV relative_energy_eV",
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "DMF calculation completed successfully in %.2f s",
        elapsed,
    )
    logger.info("=" * 70)

    return {
        "reactant_file": str(reactant_file),
        "product_file": str(product_file),
        "output_dir": str(output_dir),
        "images": mxflx.images,
        "energies": energies,
        "ts_index": ts_index,
        "ts_image": ts_image,
        "ts_energy": energies[ts_index],
        "relative_energy": relative_energy,
        "dmf": mxflx,
    }


def run_all_reactions(
    reaction_pairs,
    predictor,
    output_dir="results",
    **kwargs,
):
    """Run every reaction pair, retaining failures in the returned mapping."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    total = len(reaction_pairs)

    logger.info("=" * 70)
    logger.info("Starting batch DMF calculation")
    logger.info("Total reactions: %d", total)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 70)

    batch_start = time.perf_counter()

    for i, pair in enumerate(reaction_pairs, start=1):
        name = pair["name"]

        logger.info("")
        logger.info(
            "### Reaction %d/%d: %s ###",
            i,
            total,
            name,
        )

        reaction_start = time.perf_counter()

        try:
            results[name] = run_dmf(
                pair["reactant"],
                pair["product"],
                predictor,
                output_dir / name,
                **kwargs,
            )

            logger.info(
                "Reaction %d/%d completed: %s (%.2f s)",
                i,
                total,
                name,
                time.perf_counter() - reaction_start,
            )

        except Exception as error:
            logger.exception(
                "Reaction %d/%d FAILED: %s",
                i,
                total,
                name,
            )

            results[name] = {
                "error": error,
            }

    elapsed = time.perf_counter() - batch_start
    succeeded = sum(
        "error" not in result
        for result in results.values()
    )
    failed = total - succeeded

    logger.info("")
    logger.info("=" * 70)
    logger.info("Batch DMF calculation finished")
    logger.info("  Total    : %d", total)
    logger.info("  Succeeded: %d", succeeded)
    logger.info("  Failed   : %d", failed)
    logger.info("  Time     : %.2f s", elapsed)
    logger.info("=" * 70)

    return results
