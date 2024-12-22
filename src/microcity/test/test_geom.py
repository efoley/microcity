import jax
import jax.numpy as jnp
import pytest
import math
import numpy as np

from microcity.geom import curvature_radius, segment_tangents


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
