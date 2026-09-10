"""Which released channel a body is inverted from, and that the convex stage's recipe
produces a body in the submission pose from it."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hac26.conventions import CYLINDER_R  # noqa: E402
from hac26.data_io import (N_CAMS, load_inversion_curves, load_model_curves,  # noqa: E402
                           write_curves29)

DATA = "dataset/raw"
CKPT = "models/lpd_convex.pt"


def _write(root: Path, model: int, blender: bool, types=("intensity", "binary"), seed=0):
    """Curve files of one model under the released directory layout, with values that
    differ between the two channels so a test can tell which one was read."""
    rng = np.random.default_rng(seed + (100 if blender else 0))
    d = root / f"AsteroidModel{model:02d}_shape_secret" / f"Asteroid{model}_lightcurve_data"
    d.mkdir(parents=True, exist_ok=True)
    suffix = "_blender" if blender else ""
    for t in types:
        c = 1.0 + 0.1 * rng.standard_normal((N_CAMS, 36))
        write_curves29(str(d / f"Asteroid{model:02d}_lightcurve_{t}{suffix}.txt"),
                       np.arange(36.0), c)


def test_the_render_is_read_when_both_its_files_are_present(tmp_path):
    _write(tmp_path, 4, blender=False)
    _write(tmp_path, 4, blender=True)
    d = load_inversion_curves(str(tmp_path), 4, m=36)
    assert d["channel"] == "blender"
    assert d["mask"].sum() == 2 * N_CAMS
    ref = load_model_curves(str(tmp_path), 4, m=36, use_blender=True)
    assert np.allclose(d["curves"], ref["curves"])
    lab = load_model_curves(str(tmp_path), 4, m=36, use_blender=False)
    assert not np.allclose(d["curves"], lab["curves"])


def test_the_lab_curves_are_the_fallback_when_the_render_is_withheld(tmp_path):
    _write(tmp_path, 7, blender=False)
    d = load_inversion_curves(str(tmp_path), 7, m=36)
    assert d["channel"] == "real"
    assert d["mask"].sum() == 2 * N_CAMS
    with pytest.raises(FileNotFoundError):
        load_inversion_curves(str(tmp_path), 7, m=36, channel="blender")


def test_a_half_released_render_does_not_replace_the_lab_curves(tmp_path):
    _write(tmp_path, 8, blender=False)
    _write(tmp_path, 8, blender=True, types=("intensity",))
    d = load_inversion_curves(str(tmp_path), 8, m=36)
    assert d["channel"] == "real" and d["mask"].sum() == 2 * N_CAMS
    forced = load_inversion_curves(str(tmp_path), 8, m=36, channel="blender")
    assert forced["channel"] == "blender" and forced["mask"].sum() == N_CAMS


def test_the_channel_can_be_forced_and_must_be_named(tmp_path):
    _write(tmp_path, 9, blender=False)
    _write(tmp_path, 9, blender=True)
    d = load_inversion_curves(str(tmp_path), 9, m=36, channel="real")
    assert d["channel"] == "real"
    with pytest.raises(ValueError):
        load_inversion_curves(str(tmp_path), 9, m=36, channel="simulated")


@pytest.mark.skipif(not Path(DATA).exists() or not Path(CKPT).exists(),
                    reason="challenge data or the convex checkpoint not present")
def test_the_convex_recipe_poses_the_body_and_reads_the_render():
    from reconstruct import load_checkpoints, reconstruct_convex
    loaded = load_checkpoints([CKPT])
    v, f, info = reconstruct_convex(3, DATA, loaded)
    assert info["channel"] == "blender"
    assert abs(v[:, 2].min() + 1.0) < 1e-6 and abs(v[:, 2].max() - 1.0) < 1e-6
    r = np.sqrt((v[:, :2] ** 2).sum(1)).max()
    assert abs(r - CYLINDER_R[3]) < 1e-6
    v_lab, _, info_lab = reconstruct_convex(3, DATA, loaded, channel="real")
    assert info_lab["channel"] == "real"
    # the two channels are different recordings, so the two answers differ
    assert v_lab.shape != v.shape or not np.allclose(v_lab, v)
