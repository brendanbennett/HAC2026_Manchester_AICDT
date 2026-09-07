#!/usr/bin/env python3

# Helper functions for reconstruct_genetic.py


from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from hac26.scoring.voxel import score_mesh
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


def load_initialisation_mesh(args):

    """
    Loads a mesh for starting point of genetic algorithm

    Raises:
        FileNotFoundError: _description_
    """

    if args.initial_stl is not None:
        initial_stl = Path(args.initial_stl)

        if not initial_stl.exists():
            raise FileNotFoundError(
                f"Initial STL not found: {initial_stl}"
            )

        initial_mesh = trimesh.load(initial_stl, process=False)

        print(f"Loaded initial shape from {initial_stl}")
        print(f"  vertices: {len(initial_mesh.vertices)}")
        print(f"  faces: {len(initial_mesh.faces)}")

    else:
        initial_mesh = None

    return(initial_mesh)


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


def sh_fitness(
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


def surface_fitness(
        initial_mesh,
        params,
        influence,
        target_curves,
        cameras,
        m,
        curve_types
        ):

        mesh = deform_surface(
            initial_mesh,
            params,
            influence,
        )

        curves = mesh_curves_convex(
            mesh.vertices,
            mesh.faces,
            cameras=cameras,
            m=m,
            curve_types=curve_types,
        )

        residual = curves - target_curves

        return -np.mean(residual ** 2)


def deform_surface(
    mesh,
    amplitudes,
    influence,
):
    """
    Apply smooth signed surface deformations.

    Positive amplitude = outward bulge.
    Negative amplitude = inward indentation.
    """

    vertices = np.asarray(mesh.vertices)

    # Total displacement at each vertex
    displacement = influence @ amplitudes

    # Move along the original surface normals
    new_vertices = (
        vertices
        + displacement[:, None] * mesh.vertex_normals
    )

    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=mesh.faces.copy(),
        process=False,
    )



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


# def save_checkpoint_results(
#     output_dir,
#     args,
#     generation,
#     best_params,
#     best_fitness,
#     dice_score,
#     time_taken,
# ):
#     """Update results.json with the current checkpoint."""

#     results_path = output_dir / "results.json"

#     # Load existing results so previous checkpoints are retained
#     if results_path.exists():
#         with open(results_path, "r") as f:
#             results = json.load(f)
#     else:
#         results = {
#             "config": vars(args).copy(),
#             "checkpoints": {},
#         }

#     results["config"]["output_dir"] = str(output_dir)

#     checkpoint_name = f"generation_{generation:04d}"

#     results["checkpoints"][checkpoint_name] = {
#         "best_fitness": float(best_fitness),
#         "best_params": best_params.tolist(),
#         "dice_score": float(dice_score),
#         "time_taken": float(time_taken),
#     }

#     with open(results_path, "w") as f:
#         json.dump(
#             results,
#             f,
#             indent=2,
#         )


def save_checkpoint_results(
    output_dir,
    args,
    generation,
    best_params,
    best_fitness,
    dice_score,
    time_taken,
):
    
    """Update results.json with the current checkpoint."""

    results_path = output_dir / "results.json"

    # Load existing results so previous checkpoints are retained
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {
            "config": vars(args).copy(),
            "checkpoints": {},
        }

    # --------------------------------------------------------
    # Current best -- overwrite these every checkpoint
    # --------------------------------------------------------

    results["best_params"] = np.asarray(best_params).tolist()
    results["best_fitness"] = float(best_fitness)

    # --------------------------------------------------------
    # Checkpoint metrics -- retain history
    # --------------------------------------------------------

    if "checkpoints" not in results:
        results["checkpoints"] = {}

    results["checkpoints"][f"generation_{generation:04d}"] = {
        "dice_score": float(dice_score),
        "time_taken": float(time_taken),
        "best_fitness": float(best_fitness)
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def sample_surface_control_points(mesh, n_points, seed=42):
    """Select approximately evenly distributed vertices on a mesh."""

    rng = np.random.default_rng(seed)

    vertices = np.asarray(mesh.vertices)

    # Start from a random vertex.
    selected = [rng.integers(len(vertices))]

    # Greedy farthest-point sampling.
    min_distances = np.full(
        len(vertices),
        np.inf,
    )

    for _ in range(1, n_points):
        last = selected[-1]

        distances = np.linalg.norm(
            vertices - vertices[last],
            axis=1,
        )

        min_distances = np.minimum(
            min_distances,
            distances,
        )

        selected.append(
            np.argmax(min_distances)
        )

    return np.asarray(selected, dtype=int)

def build_surface_influence_matrix(
    mesh,
    control_point_indices,
    sigma,
):
    """
    Calculate the influence of each surface control point on
    every mesh vertex.

    Returns
    -------
    influence : ndarray
        Shape (n_vertices, n_control_points).
    """

    vertices = np.asarray(mesh.vertices)
    edges = np.asarray(mesh.edges_unique)

    # Edge lengths
    edge_lengths = np.linalg.norm(
        vertices[edges[:, 0]] - vertices[edges[:, 1]],
        axis=1,
    )

    # Build weighted mesh graph
    rows = np.concatenate(
        [edges[:, 0], edges[:, 1]]
    )

    cols = np.concatenate(
        [edges[:, 1], edges[:, 0]]
    )

    weights = np.concatenate(
        [edge_lengths, edge_lengths]
    )

    graph = coo_matrix(
        (
            weights,
            (rows, cols),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()

    # Geodesic distance from each control point to every vertex
    distances = dijkstra(
        graph,
        directed=False,
        indices=control_point_indices,
    )

    # Gaussian influence
    influence = np.exp(
        -0.5 * (distances / sigma) ** 2
    )

    # Shape:
    # distances   = (n_control_points, n_vertices)
    # influence   = (n_control_points, n_vertices)
    #
    # Transpose so that:
    # influence[vertex, control_point]
    return influence.T


## PLOTTING ## 


def plot_lightcurve_comparison(
    target_curves,
    curves_dir,
    output_path,
    m,
    plot_all=True,
):
    """Plot saved GA lightcurves against the target lightcurves.

    Parameters
    ----------
    target_curves : np.ndarray
        Truth lightcurves.

    curves_dir : Path
        Directory containing checkpoint lightcurve .npy files.

    output_path : Path
        Path for the output figure.

    m : int
        Number of phase samples.

    plot_all : bool, default=True
        If True, plot lightcurves from all saved generations.
        If False, plot only the final saved generation.
    """

    curves_dir = Path(curves_dir)

    curve_files = sorted(
        curves_dir.glob("generation_*.npy")
    )

    if not curve_files:
        raise FileNotFoundError(
            f"No lightcurve files found in {curves_dir}"
        )

    # ------------------------------------------------------------
    # Select which generations to plot
    # ------------------------------------------------------------

    if not plot_all:
        curve_files = [curve_files[-1]]

    phase = np.arange(m) / m

    # ------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------

    fig, axes = plt.subplots(
        len(target_curves),
        1,
        figsize=(8, 2 * len(target_curves)),
        sharex=True,
    )

    if len(target_curves) == 1:
        axes = [axes]

    # ------------------------------------------------------------
    # Plot truth
    # ------------------------------------------------------------

    for i, ax in enumerate(axes):
        ax.plot(
            phase,
            target_curves[i],
            label="truth",
        )

    # ------------------------------------------------------------
    # Plot GA curves
    # ------------------------------------------------------------

    for curve_file in curve_files:

        curves = np.load(curve_file)

        generation = curve_file.stem.replace(
            "generation_",
            "",
        )

        for i, ax in enumerate(axes):
            ax.plot(
                phase,
                curves[i],
                "--",
                label=f"GA {generation}",
            )

            ax.set_ylabel(f"Camera {i}")

    axes[-1].set_xlabel("Phase")

    axes[0].legend()

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