"""Generate the finer Burgers reference for the nu=0.001 resolution check."""

from scripts.dataset_generation.burgers_spectral import DATA_DIR, save_solution_for_pinn

VISCOSITY = 0.001
RESOLUTION = 4096  # Chebyshev degree: RESOLUTION + 1 spatial nodes.
FINAL_TIME = 1.0
SNAPSHOTS = 201


def main():
    """Save one candidate at the configured degree, preserving existing datasets."""
    nu_tag = f"{VISCOSITY:.8g}".replace(".", "p")
    filename = DATA_DIR / f"burgers_nu_{nu_tag}_N{RESOLUTION}.npz"
    if filename.exists():
        raise FileExistsError(f"Reference already exists: {filename}")

    # Refine only the spatial degree; match the N1024 reference's temporal settings.
    print(f"Generating nu={VISCOSITY}, N={RESOLUTION}: {filename}", flush=True)
    save_solution_for_pinn(
        nu=VISCOSITY,
        filename=filename,
        T=FINAL_TIME,
        N=RESOLUTION,
        n_snap=SNAPSHOTS,
        rtol=1e-11,
        atol=1e-13,
        max_step=0.005,
    )


if __name__ == "__main__":
    main()
