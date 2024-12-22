import jax
import jax.numpy as jnp
import pytest
import math
import numpy as np

from microcity.geom import curvature_radius, resample_polyline, segment_tangents


def test_curvature_radius_straight_line():
    # A straight line has no curvature, so we expect a very large radius.
    # Points: (0,0), (1,0), (2,0), (3,0)
    # Internal points are (1,0) and (2,0)
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    radii = curvature_radius(points)
    assert radii.shape == (2,)
    # For a perfectly straight line, we expect something very large, e.g., ~1e12
    assert (radii > 1e9).all()


def test_curvature_radius_perfect_circle():
    # Points on a circle of radius R have a curvature radius = R.
    # Let's choose a circle with radius R = 10 centered at (0,0).
    # If we pick three consecutive points on a circle of radius R, the curvature should be ~R.
    # We'll pick a small arc: angles 0°, 30°, 60°, 90°
    R = 10.0
    angles = [0, 30, 60, 90]  # degrees
    points = []
    for a in angles:
        rad = math.radians(a)
        x = R * math.cos(rad)
        y = R * math.sin(rad)
        points.append([x, y])
    points = jnp.array(points)

    # Internal points are those corresponding to angles 30° and 60°
    # We expect the curvature radius to be close to R (10.0)
    radii = curvature_radius(points)
    assert radii.shape == (2,)
    # Check if each radius is close to 10.0 within some tolerance.
    assert jnp.allclose(radii, 10.0, atol=0.1)


def test_curvature_radius_right_angle_turn():
    # A polyline that forms a right angle: points at (0,0), (1,0), (1,1), (2,1)
    # Internal points: (1,0) and (1,1)
    # The curvature at a sharp 90° turn is roughly related to the circle passing through these three points.
    # Let's just ensure it returns a finite positive radius and is not extremely large.
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    radii = curvature_radius(points)
    assert radii.shape == (2,)
    # Expect some finite positive radius
    # This won't be infinite, and it should be around ~0.5 to 1.0 depending on the configuration.
    # The exact value isn't trivial to compute mentally, but we know it should not be huge.
    assert (radii > 0).all() and (radii < 1000).all()


def test_curvature_radius_fewer_than_three_points():
    # If we provide fewer than three points, it should raise ValueError.
    points = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        _ = curvature_radius(points)


def test_curvature_radius_colinear_points():
    # Colinear but not just a straight line along x or y-axis:
    # Points along line y=x: (0,0), (1,1), (2,2), (3,3)
    # It's still a straight line, so radius should be very large.
    points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    radii = curvature_radius(points)
    assert radii.shape == (2,)
    # Check for large radius again
    assert (radii > 1e9).all()


def test_segment_tangents_straight_line():
    """
    Test that a straight polyline returns tangent vectors pointing in a single direction.
    """
    # Points along the x-axis
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    # Expected tangents: always (1, 0)
    expected = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    tangents = segment_tangents(points)
    assert tangents.shape == (3, 2)
    np.testing.assert_allclose(tangents, expected, rtol=1e-6, atol=1e-6)


def test_segment_tangents_vertical_line():
    """
    Test that a vertical polyline returns tangent vectors pointing straight up.
    """
    # Points along the y-axis
    points = jnp.array([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    # Expected tangents: (0, 1)
    expected = np.array([[0.0, 1.0], [0.0, 1.0]])

    tangents = segment_tangents(points)
    assert tangents.shape == (2, 2)
    np.testing.assert_allclose(tangents, expected, rtol=1e-6, atol=1e-6)


def test_segment_tangents_diagonal_line():
    """
    Test that a diagonal polyline of slope 1 returns uniform diagonal tangents.
    """
    # Points along line y = x
    points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    # For each segment, the direction is (1,1)/sqrt(2) => ~ (0.707..., 0.707...)
    sqrt2 = np.sqrt(2.0)
    expected = np.array([[1.0 / sqrt2, 1.0 / sqrt2], [1.0 / sqrt2, 1.0 / sqrt2]])

    tangents = segment_tangents(points)
    assert tangents.shape == (2, 2)
    np.testing.assert_allclose(tangents, expected, rtol=1e-6, atol=1e-6)


def test_segment_tangents_repeated_points():
    """
    Test handling of repeated points (zero-length segments).
    The function should still return tangents for each segment,
    with repeated points resulting in a near-zero direction vector
    that gets normalized with epsilon.
    """
    # The second and third points are identical
    points = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],  # repeated
            [2.0, 1.0],
        ]
    )
    tangents = segment_tangents(points)
    assert tangents.shape == (3, 2)

    # Check that the middle tangent won't produce NaNs
    # We expect the segment (1.0, 0.0) -> (1.0, 0.0) to be effectively 0-length,
    # so the direction is [0, 0] / eps => still [0, 0], but let's see if the
    # function handles it gracefully.
    assert not jnp.isnan(tangents).any()
    # The first segment direction ~ (1, 0)
    # The last segment direction ~ (1, 1)/sqrt(2)
    # We won't strictly test the repeated points tangent aside from ensuring no NaN/infinite.


def test_segment_tangents_random_polyline():
    """
    Sanity check on a random polyline of length N > 2.
    Just checks output shapes and ensures no NaN values.
    """
    key = jax.random.PRNGKey(42)
    # Generate 5 random points in [0, 10)
    random_points = jnp.round(10.0 * jax.random.uniform(key, shape=(5, 2)), 2)

    tangents = segment_tangents(random_points)
    assert tangents.shape == (4, 2), "Should have N-1 tangents for N points."
    assert not jnp.isnan(tangents).any(), "No NaNs expected in computed tangents."


def test_resample_polyline_straight_line():
    """
    Test that a straight line is resampled correctly with uniform spacing <= max_step.
    The original line is 4 units long, so with max_step=1.0, we expect 4 intervals of length=1.
    """
    points = jnp.array([[0.0, 0.0], [4.0, 0.0]])
    max_step = 1.0
    new_points = resample_polyline(points, max_step)

    # The line is 4 units long. We expect intervals=4, so 5 points total.
    # Positions: 0, 1, 2, 3, 4 along the x-axis.
    assert new_points.shape[0] == 5
    np.testing.assert_allclose(new_points[:, 0], [0, 1, 2, 3, 4], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(new_points[:, 1], [0, 0, 0, 0, 0], rtol=1e-6, atol=1e-6)


def test_resample_polyline_diagonal_line():
    """
    Test that a diagonal line of length sqrt(2)*3 = ~4.2426 is resampled with uniform spacing <= max_step.
    The original points are (0,0), (3,3).
    """
    points = jnp.array([[0.0, 0.0], [3.0, 3.0]])
    max_step = 1.0
    new_points = resample_polyline(points, max_step)

    # The total length is 3*sqrt(2) ~ 4.2426.
    # For max_step=1, we'd expect 5 intervals if spacing is ~0.8485 (since 4.2426 / 0.8485 ~ 5).
    # So total points = intervals + 1 = 6 (approx).
    # We won't test the exact count; let's just check the spacing is uniform and <= 1.

    distances = jnp.sqrt(jnp.sum((new_points[1:] - new_points[:-1]) ** 2, axis=1))
    # All distances should be nearly the same and <= 1.0
    assert (distances <= max_step + 1e-6).all(), "Spacing should be <= max_step"
    # Check uniformity by ensuring the std of distances is very small
    assert np.std(np.array(distances)) < 1e-6, "Distances should be uniform."


def test_resample_polyline_large_max_step():
    """
    If max_step is larger than the total length, we should only get start and end points.
    """
    points = jnp.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],  # length=5
        ]
    )
    max_step = 10.0  # larger than the entire length
    new_points = resample_polyline(points, max_step)
    # Expect just the start and end
    assert new_points.shape[0] == 2
    np.testing.assert_allclose(new_points, points, rtol=1e-6, atol=1e-6)


def test_resample_polyline_repeated_points():
    """
    If the polyline has repeated points (zero-length segments),
    ensure it doesn't cause errors or incorrect spacing in the result.
    """
    points = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],  # repeated
            [2.0, 2.0],
        ]
    )
    max_step = 0.5
    new_points = resample_polyline(points, max_step)

    # The total length ignoring repeated points = distance from (0,0)->(1,1)->(2,2) = sqrt(2)+sqrt(2) = 2*1.4142=2.8284
    # max_step=0.5 => intervals ~ ceil(2.8284 / 0.5)= ceil(5.6568)=6, spacing=2.8284/6 ~0.4714, expect ~7 points
    assert new_points.shape[0] >= 5, "Should have at least 5 points (likely more)."
    distances = jnp.sqrt(jnp.sum((new_points[1:] - new_points[:-1]) ** 2, axis=1))
    # Check spacing is uniform and <= 0.5
    assert (distances <= max_step + 1e-6).all(), "Spacing should be <= max_step"
    assert np.std(np.array(distances)) < 1e-5, "Distances should be nearly uniform."


def test_resample_polyline_random():
    """
    Quick sanity check for random input.
    Ensures no NaNs and that the output spacing is uniform and <= max_step.
    """
    key = jax.random.PRNGKey(42)
    # Generate random points in [0, 5)
    random_points = 5.0 * jax.random.uniform(key, shape=(5, 2))
    max_step = 1.0

    new_points = resample_polyline(random_points, max_step)
    assert new_points.shape[0] >= 2, "Should have at least the start and end points."
    assert not jnp.isnan(
        new_points
    ).any(), "No NaNs expected in computed resampled points."

    # Check spacing is <= max_step
    distances = jnp.sqrt(jnp.sum((new_points[1:] - new_points[:-1]) ** 2, axis=1))
    assert (distances <= max_step + 1e-6).all()
    # we don't expect spacing to be uniform here since at a corner, points may be much
    # closer than along a straight
