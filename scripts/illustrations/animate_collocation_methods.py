"""Animate the five sampler classes on a prescribed, fixed residual landscape.

Run from the repository root with
    venv/bin/python -m scripts.illustrations.animate_collocation_methods

This presentation illustration does not train a PINN or replay an experiment.
The smaller population and larger movement/noise scales make the rules visible.
"""

import copy
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from src.sampling import (
    RAD,
    FixedHammersley,
    PacmannAdam,
    PacmannDiffusion,
    RandomResampling,
)

import deepxde as dde  # isort: skip - src selects the PyTorch backend first
from deepxde import config as dde_config  # isort: skip

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
GIF_PATH = ROOT / "docs/presentation/figures/collocation_methods.gif"
PREVIEW_PATH = GIF_PATH.with_suffix(".png")
FRAMES_PATH = GIF_PATH.with_name("collocation_methods_frames.pdf")
SEED = 42
POINT_COUNT = 140
UPDATES = 60
FPS = 6
COORDINATE_STEPS = 15
STEP_SIZE = 0.002
SIGMA = 0.02
TITLES = [
    ("Fixed Hammersley", "Keep the initial points"),
    ("Random resampling", "Replace with uniform random points"),
    ("RAD", "Replace using residual-based probabilities"),
    ("PACMANN", r"Move along $\nabla_{x,t}(r^2)$ using Adam"),
    ("PACMANN + diffusion", "Add Gaussian noise to the same ascent"),
]


def residual(points, *, gradient=False):
    """Return prescribed |r|, or the analytic coordinate gradient of r squared."""
    first = points - np.array([-0.35, 0.68])
    second = points - np.array([0.42, 0.28])
    first_width = np.array([0.28, 0.20])
    second_width = np.array([0.22, 0.16])
    peak_a = np.exp(-0.5 * np.sum((first / first_width) ** 2, axis=1))
    peak_b = 0.70 * np.exp(-0.5 * np.sum((second / second_width) ** 2, axis=1))
    magnitude = 0.03 + peak_a + peak_b
    if not gradient:
        return magnitude[:, None]

    # The samplers ascend r squared, so include both the chain rule and each peak.
    derivative = -(
        peak_a[:, None] * first / first_width**2
        + peak_b[:, None] * second / second_width**2
    )
    return 2 * magnitude[:, None] * derivative


def record_dynamics():
    """Return point histories for the actual callbacks and their fixed constraints."""
    dde_config.set_default_float("float64")
    dde_config.set_random_seed(SEED)
    domain = dde.geometry.GeometryXTime(
        dde.geometry.Interval(-1, 1), dde.geometry.TimeDomain(0, 1)
    )
    bc = dde.icbc.DirichletBC(domain, lambda x: 0, lambda _, boundary: boundary)
    ic = dde.icbc.IC(domain, lambda x: 0, lambda _, initial: initial)
    data = dde.data.TimePDE(
        domain,
        residual,
        [bc, ic],
        num_domain=POINT_COUNT,
        num_boundary=24,
        num_initial=20,
        train_distribution="Hammersley",
    )
    # Match the experiment: constraints are cached separately from physics anchors.
    data.replace_with_anchors(data.train_points()[-POINT_COUNT:].copy())
    constraints = data.bc_points().copy()
    coordinate_settings: dict = {
        "interior_points": POINT_COUNT,
        "residual_gradient": partial(residual, gradient=True),
        "period": 1,
        "coordinate_steps": COORDINATE_STEPS,
        "step_size": STEP_SIZE,
        "betas": (0.9, 0.999),
        "epsilon": 1e-7,
    }
    samplers = [
        FixedHammersley(),
        RandomResampling(interior_points=POINT_COUNT, period=1),
        RAD(
            interior_points=POINT_COUNT,
            period=1,
            candidate_points=12000,
            k=1,
            c=1,
            batch_size=4096,
        ),
        PacmannAdam(**coordinate_settings),
        PacmannDiffusion(
            **coordinate_settings, sigma=SIGMA, domain_widths=(2, 1), seed=SEED
        ),
    ]

    # Supply the prescribed operators instead of a network prediction or training.
    histories = []
    for sampler in samplers:
        dde_config.set_random_seed(SEED)
        sampler.model = SimpleNamespace(
            data=copy.deepcopy(data),
            predict=lambda points, operator: operator(points),
            opt_name="adam",
        )
        frames = [sampler.model.data.train_x_all.copy()]
        for _ in range(UPDATES):
            sampler.on_epoch_end()
            frames.append(sampler.model.data.train_x_all.copy())
        histories.append(np.stack(frames))
    return histories, constraints


def main():
    """Save a sampler GIF, its final preview, and PDF frames for the slide deck."""
    histories, constraints = record_dynamics()
    x, times = np.linspace(-1, 1, 240), np.linspace(0, 1, 160)
    xx, tt = np.meshgrid(x, times)
    landscape = residual(np.column_stack((xx.ravel(), tt.ravel()))).reshape(xx.shape)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "residual", ["#fffdf6", "#fee8ac", "#fdbb6c", "#ef795c"]
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.subplots_adjust(
        left=0.055, right=0.98, bottom=0.12, top=0.79, hspace=0.50, wspace=0.23
    )
    fig.suptitle(
        "Collocation point dynamics on a fixed residual landscape",
        fontsize=19,
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Mechanism illustration with a prescribed field and illustrative parameters",
        ha="center",
        fontsize=12,
    )
    counter = fig.text(0.5, 0.875, "", ha="center", fontsize=11)

    # Every method sees the same field, axes, initial design, and fixed constraints.
    scatters = []
    for ax, history, (title, subtitle) in zip(axes.flat, histories, TITLES):
        field = ax.pcolormesh(
            x,
            times,
            landscape,
            cmap=cmap,
            vmin=0,
            vmax=1.05,
            shading="auto",
            rasterized=True,
        )
        ax.scatter(
            constraints[:, 0],
            constraints[:, 1],
            color="black",
            marker="x",
            s=14,
            linewidths=0.8,
            zorder=4,
            clip_on=False,
        )
        scatter = ax.scatter(
            history[0, :, 0], history[0, :, 1], color="black", s=10, zorder=3
        )
        scatters.append(scatter)
        ax.set(
            xlim=(-1, 1),
            ylim=(0, 1),
            xlabel="x",
            ylabel="t",
            xticks=[-1, -0.5, 0, 0.5, 1],
            yticks=[0, 0.5, 1],
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, pad=10)
        ax.set_axisbelow("line")
        ax.grid(True, linewidth=0.4, color="0.4", alpha=0.25)

    # Use the sixth panel for a shared scale and the illustration's explicit scope.
    legend_ax = axes[1, 2]
    legend_ax.set_axis_off()
    legend_ax.text(
        0,
        1.10,
        "How to read the animation.",
        transform=legend_ax.transAxes,
        fontsize=14,
        weight="bold",
    )
    legend_ax.text(
        0,
        0.92,
        r"$\mathbf{Black\ dots}$: interior collocation points."
        "\n"
        r"$\mathbf{Crosses}$: fixed initial/boundary constraints."
        "\nSame initial points in every panel.\nFixed field, no network training.",
        transform=legend_ax.transAxes,
        fontsize=11,
        linespacing=1.55,
        va="top",
    )
    legend_ax.text(
        0,
        0.43,
        f"Illustrative settings.\n{POINT_COUNT} points, {COORDINATE_STEPS} coordinate steps/update.\nStep size = {STEP_SIZE:g}, diffusion σ = {SIGMA:g}.",
        transform=legend_ax.transAxes,
        fontsize=11,
        linespacing=1.5,
        va="top",
    )
    color_ax = legend_ax.inset_axes([0, 0.01, 0.90, 0.06])
    fig.colorbar(
        field,
        cax=color_ax,
        orientation="horizontal",
        label=r"Prescribed residual magnitude $|r(x,t)|$",
    )

    def draw(update):
        """Display a recorded callback endpoint without inventing intermediate paths."""
        for scatter, history in zip(scatters, histories):
            scatter.set_offsets(history[update])
        counter.set_text(f"Sampling update {update:02d} / {UPDATES}")

    # Save a static endpoint, then render callback states with one shared palette.
    GIF_PATH.parent.mkdir(parents=True, exist_ok=True)
    draw(UPDATES)
    fig.savefig(PREVIEW_PATH, dpi=150)
    images = []
    palette = None
    with PdfPages(FRAMES_PATH) as pdf:
        for update in range(UPDATES + 1):
            draw(update)
            canvas.draw()
            frame = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3])
            if palette is None:
                palette = frame.quantize(colors=256)
            images.append(frame.quantize(palette=palette, dither=Image.Dither.NONE))

            # Repeat endpoint pages to preserve the GIF's pauses at a fixed FPS.
            repeats = 1 + (FPS if update == 0 else 2 * FPS if update == UPDATES else 0)
            for _ in range(repeats):
                pdf.savefig(fig, dpi=100)

    # A shared palette keeps the background stable and the GIF compact.
    durations = [round(1000 / FPS)] * len(images)
    durations[0] += 1000
    durations[-1] += 2000
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    plt.close(fig)
    print(f"Saved {GIF_PATH}")
    print(f"Saved {PREVIEW_PATH}")
    print(f"Saved {FRAMES_PATH}")


if __name__ == "__main__":
    main()
