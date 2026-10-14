import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import BarycentricInterpolator
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt


########################################################################
# BURGERS SOLVER
########################################################################

def make_initial_condition(x: np.ndarray) -> np.ndarray:
    """u(x, 0) = -sin(pi*x), x in [-1, 1]."""
    u0 = -np.sin(np.pi * x)

    # Enforce the Dirichlet boundary conditions exactly.
    u0[0] = 0.0
    u0[-1] = 0.0

    return u0


def chebyshev_differentiation_matrix(N: int):
    """
    Chebyshev-Gauss-Lobatto nodes and first differentiation matrix
    on [-1, 1]. N is the polynomial degree; there are N+1 nodes.
    """

    if N < 4:
        raise ValueError("N must be at least 4")
    
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.ones(N + 1)

    c[0] = 2.0
    c[-1] = 2.0
    c *= (-1.0) ** np.arange(N + 1)
    
    X = np.tile(x[:, None], (1, N + 1))
    dX = X - X.T

    D = (c[:, None] / c[None, :]) / (
    dX + np.eye(N + 1)
    )

    D -= np.diag(np.sum(D, axis=1))

    return x, D


def solve_burgers(
    nu: float,
    T: float = 1.0,
    N: int = 128,
    n_snap: int = 101,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = np.inf,
):
    """
    Solve u_t + u*u_x = nu*u_xx on [-1,1], t in [0,T],
    with u(-1,t) = u(1,t) = 0.

    N is the Chebyshev polynomial degree, giving N+1 nodes.
    max_step caps Radau's adaptive time step.

    Returns
    -------
    x : (N+1,) Chebyshev nodes, descending from 1 to -1
    t : (n_snap,) snapshot times
    U : (n_snap, N+1) solution values
    """
    if nu <= 0 or T <= 0 or N < 4 or n_snap < 2:
        raise ValueError(
            "Require nu > 0, T > 0, N >= 4, n_snap >= 2"
        )
    if rtol <= 0 or atol <= 0 or max_step <= 0:
        raise ValueError("rtol, atol and max_step must be positive")

    x, D = chebyshev_differentiation_matrix(N)
    D2 = D @ D

    u0 = make_initial_condition(x)
    t = np.linspace(0.0, T, n_snap)

    def rhs(t_now, u_interior):
        u = np.zeros(N + 1)
        u[1:-1] = u_interior

        ux = D @ u
        uxx = D2 @ u

        return (-u * ux + nu * uxx)[1:-1]

    sol = solve_ivp(
        rhs,
        (0.0, T),
        u0[1:-1],
        method="Radau",
        t_eval=t,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"Time integration failed: {sol.message}")

    U = np.zeros((n_snap, N + 1))
    U[:, 1:-1] = sol.y.T

    # Enforce boundary values exactly in the returned data.
    U[:, 0] = 0.0   # x = 1
    U[:, -1] = 0.0  # x = -1

    return x, t, U



def interpolate_to_grid(x, U, x_common):
    """
    Interpolate Chebyshev solution snapshots onto a common grid.

    x:        (N+1,) Chebyshev nodes
    U:        (n_snap, N+1)
    x_common: (n_common,) comparison points
    """
    U_common = np.empty((U.shape[0], len(x_common)))

    for j, snapshot in enumerate(U):
        interpolator = BarycentricInterpolator(x, snapshot)
        U_common[j] = interpolator(x_common)

    return U_common


def relative_l2(a, b):
    """Relative Euclidean L2 error."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    denom = np.linalg.norm(b)
    if denom == 0:
        return float(np.linalg.norm(a - b))

    return float(np.linalg.norm(a - b) / denom)


def max_abs(a, b):
    """Maximum absolute difference."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


###############################################################################
# CONVERGENCE
###############################################################################

def spatial_convergence_study(
    nu,
    N_values=(32, 64, 128, 256),
    T=1.0,
    n_snap=101,
    rtol=1e-11,
    atol=1e-13,
    reference_N=None,
):
    """
    Compare spatial resolutions against a finer Chebyshev reference.

    Errors are evaluated on a common uniform space-time grid.
    """
    N_values = sorted(set(N_values))

    if not N_values or min(N_values) < 4:
        raise ValueError("N_values must contain degrees >= 4")

    if reference_N is None:
        reference_N = 2 * max(N_values)

    if reference_N <= max(N_values):
        raise ValueError("reference_N must exceed every tested degree")

    x_common = np.linspace(-1.0, 1.0, 2 * reference_N + 1)

    x_ref, t_ref, U_ref = solve_burgers(
        nu=nu,
        T=T,
        N=reference_N,
        n_snap=n_snap,
        rtol=rtol,
        atol=atol,
    )
    U_ref_common = interpolate_to_grid(x_ref, U_ref, x_common)

    results = []

    for N in N_values:
        x, t, U = solve_burgers(
            nu=nu,
            T=T,
            N=N,
            n_snap=n_snap,
            rtol=rtol,
            atol=atol,
        )

        if not np.allclose(t, t_ref):
            raise RuntimeError("Snapshot times do not match")

        U_common = interpolate_to_grid(x, U, x_common)

        results.append({
            "nu": nu,
            "N": N,
            "reference_N": reference_N,
            "relative_L2": relative_l2(
                U_common, U_ref_common
            ),
            "max_abs_error": max_abs(
                U_common, U_ref_common
            ),
            "final_time_relative_L2": relative_l2(
                U_common[-1], U_ref_common[-1]
            ),
            "boundary_error": float(max(
                np.max(np.abs(U[:, 0])),
                np.max(np.abs(U[:, -1]))
            )),
        })

    return results



def temporal_convergence_study(
    nu,
    N=128,
    T=1.0,
    n_snap=101,
    step_counts=(10, 20, 40, 80),
    rtol=1e-12,
    atol=1e-14,
    reference_step_count=None,
):
    """
    Temporal refinement study at fixed spatial resolution.

    step_counts defines max_step = T / count.
    An independent, finer solution is used as the reference.
    """
    step_counts = sorted(set(step_counts))

    if not step_counts or min(step_counts) < 1:
        raise ValueError("step_counts must be positive integers")

    if reference_step_count is None:
        reference_step_count = 4 * max(step_counts)

    if reference_step_count <= max(step_counts):
        raise ValueError(
            "reference_step_count must exceed every tested count"
        )

    # Compute each tested solution.
    solutions = {}

    for count in step_counts:
        x, t, U = solve_burgers(
            nu=nu,
            T=T,
            N=N,
            n_snap=n_snap,
            rtol=rtol,
            atol=atol,
            max_step=T / count,
        )
        solutions[count] = (x, t, U)

    # Independent reference with a smaller maximum step and
    # tighter solver tolerances.
    ref_rtol = min(rtol, 1e-11)
    ref_atol = min(atol, 1e-13)

    x_ref, t_ref, U_ref = solve_burgers(
        nu=nu,
        T=T,
        N=N,
        n_snap=n_snap,
        rtol=ref_rtol,
        atol=ref_atol,
        max_step=T / reference_step_count,
    )

    results = []

    for count in step_counts:
        x, t, U = solutions[count]

        if not np.allclose(t, t_ref):
            raise RuntimeError("Snapshot times do not match")

        # Direct comparison diagnostic.
        difference = U - U_ref
        max_difference = float(np.max(np.abs(difference)))

        relative_error = relative_l2(U, U_ref)
        final_time_error = relative_l2(U[-1], U_ref[-1])

        results.append({
            "nu": nu,
            "N": N,
            "step_count": count,
            "max_step": T / count,
            "reference_step_count": reference_step_count,
            "relative_L2": relative_error,
            "max_abs_error": max_difference,
            "final_time_relative_L2": final_time_error,
        })


        if max_difference == 0.0:
            print(
                "  WARNING: tested solution is identical to the "
                "reference at the saved points."
            )

    return results




def save_results_csv(results, filename):
    if not results:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)


def plot_convergence_results(
    spatial_results,
    temporal_results,
    output_dir="convergence_results",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Spatial convergence
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    spatial_viscosities = sorted({
        row["nu"] for row in spatial_results
    })

    for nu in spatial_viscosities:
        rows = sorted(
            [r for r in spatial_results if r["nu"] == nu],
            key=lambda r: r["N"],
        )

        N_values = [r["N"] for r in rows]
        errors = [r["relative_L2"] for r in rows]

        ax.loglog(
            N_values,
            errors,
            marker="o",
            label=fr"$\nu={nu:.4g}$",
        )

    ax.set_xlabel("Chebyshev polynomial degree N")
    ax.set_ylabel("Relative space-time L2 error")
    ax.set_title("Spatial convergence")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "spatial_L2_convergence.png",
        dpi=200,
        bbox_inches="tight",
    )

    # ---------------------------------------------------------
    # Temporal convergence
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    temporal_viscosities = sorted({
        row["nu"] for row in temporal_results
    })

    for nu in temporal_viscosities:
        rows = sorted(
            [r for r in temporal_results if r["nu"] == nu],
            key=lambda r: r["max_step"],
        )

        # Logarithmic axes cannot display zero errors.
        positive_rows = [
            r for r in rows if r["relative_L2"] > 0
        ]

        if positive_rows:
            ax.loglog(
                [r["max_step"] for r in positive_rows],
                [r["relative_L2"] for r in positive_rows],
                marker="o",
                label=fr"$\nu={nu:.4g}$",
            )
        else:
            print(
                f"WARNING: no positive temporal L2 errors "
                f"to plot for nu={nu:.6g}."
            )

    ax.set_xlabel("Maximum time step")
    ax.set_ylabel("Relative space-time L2 error")
    ax.set_title("Temporal convergence")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_dir / "temporal_L2_convergence.png",
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nPlots saved in: {output_dir}")
    plt.show()


def convergence_studies_over_viscosities(
    viscosities,
    output_dir="convergence_results",
    spatial_N_values=(32, 64, 128, 256),
    spatial_reference_N=512,
    temporal_N=128,
    temporal_step_counts=(10, 20, 40, 80, 160),
    T=1.0,
    n_snap=101,
    rtol=1e-11,
    atol=1e-13,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_spatial = []
    all_temporal = []

    for nu in viscosities:
        print(f"\nConvergence studies for nu={nu:.8g}")

        spatial = spatial_convergence_study(
            nu=nu,
            N_values=spatial_N_values,
            T=T,
            n_snap=n_snap,
            rtol=rtol,
            atol=atol,
            reference_N=spatial_reference_N,
        )
        all_spatial.extend(spatial)

        temporal = temporal_convergence_study(
            nu=nu,
            N=temporal_N,
            T=T,
            n_snap=n_snap,
            step_counts=temporal_step_counts,
            rtol=rtol,
            atol=atol,
        )
        all_temporal.extend(temporal)

    save_results_csv(
        all_spatial,
        output_dir / "spatial_convergence.csv",
    )
    save_results_csv(
        all_temporal,
        output_dir / "temporal_convergence.csv",
    )

    print(f"\nConvergence results saved in: {output_dir}")

    plot_convergence_results(
        spatial_results=all_spatial,
        temporal_results=all_temporal,
        output_dir=output_dir,
    )

    return all_spatial, all_temporal

####################################################################################
# SOLVE CASE FOR PINN DATASET
####################################################################################


def save_solution_for_pinn(
    nu,
    filename,
    T=1.0,
    N=256,
    n_snap=201,
    rtol=1e-11,
    atol=1e-13,
    max_step=np.inf,
):
    """
    Generate and save one Burgers solution for PINN training/validation.

    Saved fields:
      x, t       : spatial nodes and time snapshots
      U          : shape (n_snap, N+1), U[j,i] = u(x[i], t[j])
      X_flat     : flattened spatial coordinates
      T_flat     : flattened time coordinates
      u_flat     : flattened solution values
      nu, config : viscosity and solver settings
    """
    x, t, U = solve_burgers(
        nu=nu,
        T=T,
        N=N,
        n_snap=n_snap,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    # Mesh arrays have shape (n_snap, N+1).
    X, TT = np.meshgrid(x, t)

    config = {
        "equation": "u_t + u*u_x = nu*u_xx",
        "domain_x": [-1.0, 1.0],
        "domain_t": [0.0, float(T)],
        "initial_condition": "-sin(pi*x)",
        "boundary_conditions": "u(-1,t)=u(1,t)=0",
        "nu": float(nu),
        "N": int(N),
        "number_of_spatial_nodes": int(N + 1),
        "n_snap": int(n_snap),
        "rtol": float(rtol),
        "atol": float(atol),
        "max_step": float(max_step),
        "time_integrator": "Radau",
        "spatial_method": "Chebyshev collocation",
    }

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        filename,
        x=x,
        t=t,
        U=U,
        X_flat=X.ravel(),
        T_flat=TT.ravel(),
        u_flat=U.ravel(),
        nu=np.float64(nu),
        config_json=np.array(json.dumps(config)),
    )

    print(f"Saved: {filename}")
    print(f"U shape: {U.shape}")
    print(f"Flattened samples: {U.size}")

    return x, t, U

def generate_viscosity_sweep(
    viscosities,
    output_dir="burgers_dataset",
    T=1.0,
    N=256,
    n_snap=201,
    rtol=1e-11,
    atol=1e-13,
    max_step=np.inf,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for nu in viscosities:
        # Use a readable, unique filename for each viscosity.
        nu_tag = f"{nu:.8g}".replace(".", "p")
        filename = output_dir / f"burgers_nu_{nu_tag}.npz"

        save_solution_for_pinn(
            nu=nu,
            filename=filename,
            T=T,
            N=N,
            n_snap=n_snap,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

####################################################################################
####################################################################################


if __name__ == "__main__":

    nu = 0.01 / np.pi
    T = 1.0
    N = 1024
    n_snap = 201

    rtol = 1e-11
    atol = 1e-13
    max_step = 0.005

    # Output configuration
    output_dir = Path("burgers_dataset")
    nu_tag = f"{nu:.8g}".replace(".", "p")
    filename = output_dir / f"burgers_nu_{nu_tag}_N{N}.npz"

    # =========================================================
    # SOLVE AND SAVE
    # =========================================================

    x, t, U = save_solution_for_pinn(
        nu=nu,
        filename=filename,
        T=T,
        N=N,
        n_snap=n_snap,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

#######################
# CONVERGENCE
#######################

    # viscosities = [0.01 / np.pi]

    # spatial_results, temporal_results = (
    #     convergence_studies_over_viscosities(
    #         viscosities=viscosities,
    #         output_dir="convergence_results",
    #         spatial_N_values=(64, 128, 512),
    #         spatial_reference_N=1024,
    #         temporal_N=100,
    #         temporal_step_counts=(10, 50),
    #         T=1.0,
    #         n_snap=21,
    #         rtol=1e-9,
    #         atol=1e-11,
    #     )
    # )

    # print("\nSpatial convergence results:")
    # for result in spatial_results:
    #     print(result)

    # print("\nTemporal convergence results:")
    # for result in temporal_results:
    #     print(result)

    



