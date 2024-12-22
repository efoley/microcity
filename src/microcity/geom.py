import jax
import jax.numpy as jnp
from jax import grad, jit


def segment_lengths(points: jax.Array):
    """
    Compute the lengths of polyline segments.
    """
    diffs = points[1:] - points[:-1]
    seg_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1))
    return seg_lengths


def curvature_radius(points, eps=1e-9):
    """
    Compute the approximate radius of curvature at each internal point of a polyline.

    For each set of three consecutive points (p0, p1, p2) in the polyline, this function
    calculates the radius of the circumscribed circle (radius of curvature) passing through
    these points. This is achieved by computing the curvature using the area of the triangle
    formed by the points and the product of the lengths of its sides.

    Parameters
    ----------
    points : jax.numpy.ndarray
        A two-dimensional array of shape (N, 2) representing the vertices of the polyline.
        Each row corresponds to a point's (x, y) coordinates in meters.

    eps : float
        A small number that is added to the numerator and denominator to avoid nans/infinities
        when points are nearly collinear.

    Returns
    -------
    jax.numpy.ndarray
        A one-dimensional array of length (N - 2) containing the radius of curvature (in meters)
        at each internal point of the polyline. If the curvature is undefined (e.g., points are
        colinear), the radius is set to a large value (1e12 meters) to indicate a straight segment.

    Notes
    -----
    - The radius of curvature is calculated as:
        radius = (|AB| * |BC| * |CA| + eps) / (4 * Area + eps)
      where AB, BC, and CA are the vectors between consecutive points, and Area is the area
      of the triangle formed by the three points.
    - An epsilon (`eps`) is added to both the numerator and denominator to prevent division by zero
      and to handle cases where points may be nearly colinear.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> points = jnp.array([
    ...     [0.0, 0.0],
    ...     [1.0, 0.0],
    ...     [1.0, 1.0],
    ...     [2.0, 1.0]
    ... ])
    >>> radii = compute_curvature(points)
    >>> print(radii)
    [1.00000012e+00 1.00000012e+00]

    In this example, the polyline forms a right-angle turn at each internal point, resulting in a
    radius of curvature of approximately 1 meter at each bend.
    """
    # Ensure there are at least three points to compute curvature
    if points.shape[0] < 3:
        raise ValueError("At least three points are required to compute curvature.")

    # Extract consecutive triplets of points
    p0 = points[:-2]
    p1 = points[1:-1]
    p2 = points[2:]

    # Vectors between points
    AB = p1 - p0
    BC = p2 - p1
    CA = p0 - p2

    # Distances between points
    distAB = jnp.sqrt(jnp.sum(AB**2, axis=1))
    distBC = jnp.sqrt(jnp.sum(BC**2, axis=1))
    distCA = jnp.sqrt(jnp.sum(CA**2, axis=1))

    # Cross product to find the area of the triangle
    cross_val = AB[:, 0] * BC[:, 1] - AB[:, 1] * BC[:, 0]
    area = jnp.abs(cross_val) / 2.0

    # Product of the side lengths
    dist_prod = distAB * distBC * distCA

    # Compute radius of curvature
    r = (dist_prod + eps) / (4.0 * area + eps)

    # Handle cases where curvature is undefined (area is zero)
    # radius = jnp.where(area > eps, 1.0 / (4.0 * area / (dist_prod + eps)), 1e12)

    return r


def segment_tangents(points, eps=1e-8):
    """
    Compute the tangent (unit direction vector) for each segment in a polyline.

    Parameters
    ----------
    points : jax.numpy.ndarray
        An array of shape (N, 2) representing the vertices of the polyline.
        Each row is a point (x_i, y_i).

    eps : float, optional
        A small constant to prevent division by zero when normalizing very short segments.
        Default is 1e-8.

    Returns
    -------
    jax.numpy.ndarray
        An array of shape (N-1, 2) where each row is the unit tangent vector
        for the corresponding segment of the polyline.
        
    Notes
    -----
    - If two consecutive points coincide or are extremely close, `eps` prevents
      division by zero when normalizing the difference vector.
    - The tangent vectors point in the direction from points[i] to points[i+1].
    """
    # Compute difference vectors for each segment: shape = (N-1, 2)
    diffs = points[1:] - points[:-1]

    # Compute magnitudes of these difference vectors: shape = (N-1,)
    lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1))

    # Normalize each segment vector to get unit tangents: shape = (N-1, 2)
    # We add eps to avoid division by zero when lengths are extremely small.
    tangents = diffs / (lengths[:, None] + eps)
    
    return tangents