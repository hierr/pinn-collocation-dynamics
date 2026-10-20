"""Check the illustration's analytic score gradient and shared point controls."""

import numpy as np
from PIL import Image, ImageSequence

from scripts import animate_collocation_methods as animation


def test_squared_residual_gradient():
    """Match the coordinate derivative of r squared against central differences."""
    points = np.array([[-0.6, 0.5], [0.2, 0.3], [0.0, 0.8]])
    expected = np.empty_like(points)
    step = 1e-6
    for coordinate in range(2):
        offset = np.zeros_like(points)
        offset[:, coordinate] = step
        expected[:, coordinate] = (
            (
                animation.residual(points + offset) ** 2
                - animation.residual(points - offset) ** 2
            )
            / (2 * step)
        ).ravel()
    np.testing.assert_allclose(
        animation.residual(points, gradient=True),
        expected,
        rtol=1e-8,
        atol=1e-9,
    )


def test_matched_initial_points_and_domain(monkeypatch):
    """Keep fixed sampling unchanged and all animated populations inside the domain."""
    monkeypatch.setattr(animation, "POINT_COUNT", 32)
    monkeypatch.setattr(animation, "UPDATES", 3)
    histories, constraints = animation.record_dynamics()
    assert len(histories) == 5
    assert len(constraints) > 0
    for history in histories:
        assert history.shape == (4, 32, 2)
        np.testing.assert_array_equal(history[0], histories[0][0])
        assert np.isfinite(history).all()
        assert ((history[:, :, 0] >= -1) & (history[:, :, 0] <= 1)).all()
        assert ((history[:, :, 1] >= 0) & (history[:, :, 1] <= 1)).all()
    np.testing.assert_array_equal(histories[0], np.repeat(histories[0][:1], 4, axis=0))
    for history in histories[1:]:
        assert not np.array_equal(history[0], history[-1])


def test_animation_outputs(tmp_path, monkeypatch):
    """Render a looping GIF, static preview, and PDF frames with a tiny budget."""
    monkeypatch.setattr(animation, "POINT_COUNT", 24)
    monkeypatch.setattr(animation, "UPDATES", 2)
    monkeypatch.setattr(animation, "FPS", 2)
    gif = tmp_path / "methods.gif"
    preview = tmp_path / "methods.png"
    frames = tmp_path / "methods_frames.pdf"
    monkeypatch.setattr(animation, "GIF_PATH", gif)
    monkeypatch.setattr(animation, "PREVIEW_PATH", preview)
    monkeypatch.setattr(animation, "FRAMES_PATH", frames)
    animation.main()
    with Image.open(gif) as image:
        assert sum(1 for _ in ImageSequence.Iterator(image)) >= 3
        assert image.info["loop"] == 0
    with Image.open(preview) as image:
        assert image.format == "PNG"
    assert frames.read_bytes().startswith(b"%PDF-")
