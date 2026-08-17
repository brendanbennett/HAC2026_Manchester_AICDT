#!/usr/bin/env python3

# Helper functions for reconstruct_genetic.py


from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import trimesh


from hac26.shapes import (
    icosphere,
    sh_mesh_from_coefficients,
    mesh_curves_convex,
)


import trimesh


def load_truth_mesh(model, data_dir):
    """Load a public challenge asteroid mesh."""

    truth_files = {
        1: "AsteroidModel01_shape_public/asteroid1.stl",
        2: "AsteroidModel02_shape_public/asteroid2.stl",
        3: "AsteroidModel03_shape_public/asteroid3.stl",
    }

    if model not in truth_files:
        raise ValueError(
            f"No public truth shape available for model {model}."
        )

    path = data_dir / truth_files[model]

    if not path.exists():
        raise FileNotFoundError(
            f"Truth STL not found: {path}"
        )

    return trimesh.load(
        path,
        process=False,
    )


def make_target_coefficients(
    rng: np.random.Generator,
    L: int,
    amp: float = 0.35,
    decay: float = 1.5,
) -> np.ndarray:
    """Generate the hidden SH coefficients for the toy target."""

    ls = np.concatenate(
        [
            [l] * (2 * l + 1)
            for l in range(1, L + 1)
        ]
    )

    return rng.normal(
        0.0,
        amp / (1.0 + ls) ** decay,
    )


def shape_fitness(
    coefficients: np.ndarray,
    target_coefficients: np.ndarray,
) -> float:
    """Toy fitness based on coefficient recovery.

    This is deliberately simple for the first test. The target coefficients
    are hidden from the GA, but the fitness currently measures distance in
    coefficient space.

    This will later be replaced by a lightcurve-based fitness.
    """

    error = np.sum(
        (coefficients - target_coefficients) ** 2
    )

    return -error


def lightcurve_fitness(
    coefficients,
    target_curves,
    L,
    subdiv,
    cameras,
    m,
    curve_types,
    c_lambert=0.1,
    sigma=1.0,
    delta=1.0,
    psi0=0.0,
    ls_weight=1.0,
    ):
    """Calculate fitness by comparing model and target lightcurves.

    Higher fitness is better, so this returns the negative mean squared
    lightcurve residual.
    """

    # ------------------------------------------------------------
    # Candidate shape
    # ------------------------------------------------------------

    vertices, faces = sh_mesh_from_coefficients(
        coefficients,
        L=L,
        subdiv=subdiv,
    )

    # ------------------------------------------------------------
    # Candidate lightcurves
    # ------------------------------------------------------------

    curves = mesh_curves_convex(
        vertices,
        faces,
        cameras=cameras,
        m=m,
        curve_types=curve_types,
        c_lambert=c_lambert,
        sigma=sigma,
        delta=delta,
        psi0=psi0,
        ls_weight=ls_weight,
    )

    # ------------------------------------------------------------
    # Residual
    # ------------------------------------------------------------

    residual = curves - target_curves

    # Mean squared error over all cameras and phases.
    mse = np.mean(residual**2)

    return -mse


def plot_mesh(
    ax,
    vertices,
    faces,
    title,
):
    """Plot a triangular mesh using matplotlib."""

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        linewidth=0.1,
        alpha=0.9,
    )

    ax.set_title(title)
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def plot_mesh_views(
    vertices,
    faces,
    title,
    axes,
):
    """Plot six views of a triangular mesh."""

    views = [
        (90, -90, "XY"),
        (0, -90, "XZ"),
        (0, 0, "YZ"),
        (30, 45, "Iso +++"),
        (30, 135, "Iso -++"),
        (30, -45, "Iso +-+"),
    ]

    for ax, (elev, azim, view_title) in zip(
        axes,
        views,
    ):
        ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces,
            linewidth=0.1,
            alpha=0.9,
        )

        ax.view_init(
            elev=elev,
            azim=azim,
        )

        ax.set_title(view_title)

        ax.set_box_aspect(
            (1, 1, 1)
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def save_shape_stl(
    coefficients,
    L,
    subdiv,
    path,
):
    """Generate an SH mesh and save it as an STL file."""

    vertices, faces = sh_mesh_from_coefficients(
        coefficients,
        L=L,
        subdiv=subdiv,
    )

    
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh.export(path)

    print(f"Saved {path}")

    return mesh


def save_results_json(
    output_dir,
    args,
    result,
    dice_scores,
    comp_t
):
    """Save run configuration and optimisation results to JSON."""


    def _json_serializable(obj):
        """Convert common NumPy types to JSON-serializable Python types."""

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.bool_):
            return bool(obj)

        if isinstance(obj, dict):
            return {
                key: _json_serializable(value)
                for key, value in obj.items()
            }

        if isinstance(obj, (list, tuple)):
            return [
                _json_serializable(value)
                for value in obj
            ]

        return obj

    config = vars(args).copy()
    config["n_coefficients"] = args.L * (args.L + 2)
    config["output_dir"] = str(output_dir)

    results = {
        "config": config,
        "result": {
            "dice_scores": dice_scores,
            "best_fitness": float(result.best_fitness),
            "best_params": result.best_params.tolist(),
            "fitness_history": [
                float(x)
                for x in result.best_fitness_history
            ],
        },
        "timing": {
            "total_seconds": comp_t,
        },
    }


    with open(output_dir / "results.json", "w") as f:
        json.dump(
            _json_serializable(results),
            f,
            indent=2,
        )



## PLOTTING ## 

def plot_lightcurve_comparison(
    vertices,
    faces,
    target_curves,
    cameras,
    curve_types,
    m,
    output_path,
):
    """Plot target and reconstructed lightcurves for each camera.

    Parameters
    ----------
    vertices, faces
        Mesh of the reconstructed shape.

    target_curves : np.ndarray
        Target lightcurves, shape (n_cameras, m).

    cameras : list
        Camera definitions used by the forward model.

    curve_types : list
        Curve type for each camera.

    m : int
        Number of phase samples.

    output_path : str or Path
        Path at which to save the figure.
    """

    final_curves = mesh_curves_convex(
        vertices,
        faces,
        cameras=cameras,
        m=m,
        curve_types=curve_types,
    )

    fig, axes = plt.subplots(
        len(cameras),
        1,
        figsize=(8, 2 * len(cameras)),
        sharex=True,
    )

    if len(cameras) == 1:
        axes = [axes]

    phase = np.arange(m) / m

    for i, ax in enumerate(axes):

        ax.plot(
            phase,
            target_curves[i],
            label="truth",
        )

        ax.plot(
            phase,
            final_curves[i],
            "--",
            label="GA",
        )

        ax.set_ylabel(f"Camera {i}")

        ax.legend()

    axes[-1].set_xlabel("Phase")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)


def plot_genetic_convergence(
    best_fitness_history,
    output_path,
):
    """Plot genetic algorithm convergence.

    Parameters
    ----------
    best_fitness_history : array-like
        Best fitness recorded at each generation.

    output_path : str or Path
        Path at which to save the figure.
    """

    generations = np.arange(len(best_fitness_history))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        generations,
        best_fitness_history,
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.set_title("Genetic algorithm convergence")

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(fig)   