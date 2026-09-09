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
    solid_centroid,
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


class ExactForwardModel:
    """The flow-matching (LPD) stage's own forward model
    (hac26.forward.mesh.exact.ExactForward), wrapped to the same (verts, faces, cameras, m,
    curve_types) -> (n_curves, m) contract mesh_curves_convex already has, so it drops into
    render_curves/sh_fitness/surface_fitness/target-curve generation unchanged.

    Why this exists: mesh_curves_convex (hac26.forward.convex_egi.kernel) gives every facet's
    brightness from its own normal alone -- mu > 0 and mu0 > 0 -- with no visibility test
    against any other facet on the body, so it cannot represent cast shadows or
    interreflected light. That is exactly right for a convex candidate (a convex body has no
    self-occlusion to model) but increasingly wrong as a candidate becomes concave, which is
    the entire point of the surface-deformation GA. ExactForward instead rasterises real cast
    shadows and solves a radiosity system for interreflection between faces -- what the flow
    stage is actually trained against.

    That fidelity costs real compute: expect at least an order of magnitude slower per
    candidate than mesh_curves_convex, more on a machine with no CUDA/nvdiffrast, which falls
    back to the pure-torch software rasteriser (hac26.forward.mesh.raster._backend) -- this
    repo documents that backend as a stand-in for tests, not for real runs. Construction loads
    the calibrated Instrument and builds its Rasterisers, both expensive; build one instance
    and reuse it across every candidate and generation, never per call.

    Camera correspondence: `cameras` passed to .curves() are matched to
    hac26.conventions.cameras() purely by list position, not by re-deriving azimuth/elevation.
    hac26.geometry.build_cameras() (what the GA otherwise uses) and hac26.conventions.cameras()
    independently hardcode the same released azimuths/elevations in the same per-azimuth
    (hor_a, hor_b, top, bottom) order -- the two lists agree index for index, which is the
    invariant hac26.conventions's own Camera docstring already relies on. Passing any other
    camera list here is not supported.
    """

    def __init__(self, calibration: str, m: int, device: str = "cpu", backend: str | None = None,
                radiosity_faces: int = 200, form_factor_samples: int = 4,
                height: int = 108, width: int = 192):
        import torch
        from hac26.conventions import cameras as conv_cameras
        from hac26.conventions import psi_grid as conv_psi_grid
        from hac26.forward.mesh.exact import ExactForward, RenderConfig
        from hac26.forward.mesh.instrument import Instrument

        self._torch = torch
        self.device = device
        inst = Instrument.load(calibration, device=device)
        cfg = RenderConfig(height=height, width=width, radiosity_faces=radiosity_faces,
                           form_factor_samples=form_factor_samples)
        psi = conv_psi_grid(frames=m)
        self._forward = ExactForward(inst, psi, config=cfg, device=device, backend=backend)
        self._n_cams = len(conv_cameras())
        # hac26.geometry.build_cameras() returns fresh Camera objects on every call (a
        # dataclass, but a new instance each time), so matching by object identity would
        # never work; Camera is frozen (value-equal and hashable), so a dict built once from
        # one reference call is a correct and cheap way to turn a caller's Camera back into
        # its position in that list -- built here rather than per .curves() call.
        from hac26.geometry import build_cameras
        self._cam_index = {cam: i for i, cam in enumerate(build_cameras())}

    def curves(self, verts: np.ndarray, faces: np.ndarray, cameras: list,
              curve_types: list) -> np.ndarray:
        """(n_curves, m): one row per (cameras[i], curve_types[i]) pair, matching
        mesh_curves_convex's contract. `cameras` must be hac26.geometry.build_cameras()'s own
        list (or a subset of it, by position) -- see the class docstring."""
        torch = self._torch
        v = torch.as_tensor(np.ascontiguousarray(verts, dtype=np.float32), device=self.device)
        f = torch.as_tensor(np.ascontiguousarray(faces, dtype=np.int64), device=self.device)
        geoms = list(range(self._n_cams))
        with torch.no_grad():
            raw = self._forward.raw_curves(v, f, geoms=geoms)          # (G, 2, m), no grad
        raw = raw.cpu().numpy()
        kind = {"intensity": 0, "binary": 1}
        try:
            rows = [raw[self._cam_index[cam], kind[ctype]] for cam, ctype in
                   zip(cameras, curve_types)]
        except KeyError as exc:
            raise ValueError("camera not in hac26.geometry.build_cameras() -- ExactForwardModel "
                             "only supports the GA's own fixed camera list, see the class "
                             "docstring") from exc
        return np.stack(rows, axis=0)


def render_curves(verts: np.ndarray, faces: np.ndarray, cameras: list, m: int,
                  curve_types: list, forward: "ExactForwardModel | None" = None,
                  **convex_kwargs) -> np.ndarray:
    """Curves of a candidate mesh: mesh_curves_convex by default, or `forward.curves(...)`
    (an ExactForwardModel, the flow stage's own forward model) when one is given. Both return
    the same (n_curves, m) shape, so this is what sh_fitness/surface_fitness and the
    target-curve generation in scripts/reconstruct_genetic.py and
    scripts/tune_genetic_hyperparams.py call, and which model they use only depends on
    whether a `forward` was built and threaded through -- nothing else about the GA changes."""
    if forward is None:
        return mesh_curves_convex(verts, faces, cameras=cameras, m=m, curve_types=curve_types,
                                  **convex_kwargs)
    return forward.curves(verts, faces, cameras, curve_types)


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
    forward=None,
    ):
    """Calculate fitness by comparing model and target lightcurves.

    Higher fitness is better, so this returns the negative mean squared
    lightcurve residual. `forward` selects the forward model (render_curves); c_lambert,
    sigma, delta, psi0, ls_weight only apply to the default convex one.
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

    convex_kwargs = {} if forward is not None else dict(
        c_lambert=c_lambert, sigma=sigma, delta=delta, psi0=psi0, ls_weight=ls_weight)
    curves = render_curves(
        vertices,
        faces,
        cameras=cameras,
        m=m,
        curve_types=curve_types,
        forward=forward,
        **convex_kwargs,
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
        curve_types,
        forward=None,
        ):

        mesh = deform_surface(
            initial_mesh,
            params,
            influence,
        )

        curves = render_curves(
            mesh.vertices,
            mesh.faces,
            cameras=cameras,
            m=m,
            curve_types=curve_types,
            forward=forward,
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

    Vertices move along the radial direction from the mesh's solid centroid, not along
    per-vertex surface normals. A surface normal is a function of the local triangle
    geometry, so at a high-curvature region (an elongated body's tips) neighbouring
    vertices can have sharply diverging normals; pushing them along those diverging
    directions is what produced the spiky/jagged artefacts seen at those tips. The radial
    direction is a smooth function of vertex position alone, so it has no such divergence.
    """

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    centre = solid_centroid(vertices, faces)
    radial = vertices - centre
    radial_unit = radial / np.linalg.norm(radial, axis=1, keepdims=True)

    # Total displacement at each vertex
    displacement = influence @ amplitudes

    # Move along the radial direction from the body's centre
    new_vertices = (
        vertices
        + displacement[:, None] * radial_unit
    )

    return trimesh.Trimesh(
        vertices=new_vertices,
        faces=faces.copy(),
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