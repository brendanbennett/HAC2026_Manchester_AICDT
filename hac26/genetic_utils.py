#!/usr/bin/env python3

# Helper functions for reconstruct_genetic.py

import argparse
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import trimesh
import time

from hac26.shapes import (
    icosphere,
    sh_mesh_from_coefficients,
    mesh_curves_convex,
)
from hac26.solvers.genetic import GeneticSolver
from hac26.geometry import build_cameras
from hac26.recon import dice
from hac26.scoring.voxel import score_mesh, prepare_truth


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


    config = vars(args).copy()
    config["n_coefficients"] = args.L * (args.L + 2)

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
            results,
            f,
            indent=2,
        )