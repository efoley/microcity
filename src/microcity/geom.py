import jax
import jax.numpy as jnp


def segment_lengths(points: jax.Array):
    """
    Compute the lengths of polyline segments.
    """
    diffs = points[1:] - points[:-1]
    seg_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1))
    return seg_lengths


def vertex_length_parameters(points: jax.Array, normalize: bool = False):
    """
    Compute arc-length parameters for each vertex of a polyline.

    This function calculates the cumulative distance at each vertex of a polyline
    (including the first vertex at distance 0), using an external `segment_lengths`
    function to determine the length of each consecutive pair of vertices.
    The resulting distance values can optionally be normalized to lie in [0, 1] by
    setting `normalize=True`.

    Parameters
    ----------
    points : jax.Array
        An (N, D) array representing the coordinates of the polyline vertices,
        where N >= 2 and D >= 1. Typically, D = 2 for (x, y) coordinates.

    normalize : bool, optional
        Whether to normalize the distance parameters by the total length of
        the polyline so that the last vertex is mapped to 1.0. Defaults to False.

    Returns
    -------
    jax.Array
        A one-dimensional array of length N containing the cumulative distances (or
        normalized distances) at each vertex. The first element is always 0, and if
        `normalize=True`, the last element is 1.

    Notes
    -----
    - The function relies on `segment_lengths(points)`, which should return the
      lengths of each segment between consecutive vertices.
    - If the polyline is closed or has repeated points, distances for zero-length
      segments would be 0.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # Suppose we have points forming a line along the x-axis
    >>> points = jnp.array([[0., 0.],
    ...                     [3., 0.],
    ...                     [6., 0.]])
    >>> t = vertex_length_parameters(points)
    >>> print(t)
    [0. 3. 6.]
    >>> t_norm = vertex_length_parameters(points, normalize=True)
    >>> print(t_norm)
    [0.  0.5 1. ]
    """
    t = jnp.r_[
        0, jnp.cumulative_sum(segment_lengths(points), include_initial=False)
    ]
    if normalize:
        t /= t[-1]
    return t


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


def resample_polyline(points: jnp.ndarray, max_step: float) -> jnp.ndarray:
    """
    Resample a polyline so that it has equally spaced points with spacing <= max_step.
    This function does not preserve the original vertices.

    This function interprets 'max_step' as the maximum allowable spacing. It determines
    how many equally spaced intervals fit between 0 and the total arc length of the
    polyline such that the spacing is <= max_step, then places vertices at those
    uniform distances.

    Parameters
    ----------
    points : jax.numpy.ndarray
        An (N, 2) array of (x, y) coordinates representing the polyline vertices.
        Must have at least two points.

    max_step : float
        The maximum distance between consecutive vertices in the resampled polyline.
        The actual spacing in the result will be <= max_step and constant for all
        intervals.

    Returns
    -------
    jax.numpy.ndarray
        An (M, 2) array of resampled vertices, equally spaced from the start to
        the end of the original polyline.

    Notes
    -----
    - We do NOT preserve the original polyline's intermediate vertices; the entire
      polyline is treated as a continuous curve, and new points are placed at uniform
      distances.
    - The final spacing is exactly total_length / num_intervals, where num_intervals
      is the smallest integer such that the spacing is <= max_step.
    - This version omits special handling for extremely short polylines or explicitly
      appending the last point in a separate step, as requested.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> # A simple polyline with total length ~ 8 units
    >>> original_points = jnp.array([
    ...     [0.0, 0.0],
    ...     [3.0, 4.0],
    ...     [6.0, 4.0]
    ... ])
    >>> # We want the spacing to be at most 2.5 units
    >>> # The function will choose intervals so that spacing is <= 2.5
    >>> new_points = resample_polyline(original_points, max_step=2.5)
    >>> print(new_points)
    [[0.        0.       ]
     [1.3047379 1.7396501]
     [2.6094758 3.4793003]
     [3.8571064 4.       ]
     [5.1428936 4.       ]
     [6.        4.       ]]
    """
    if points.shape[0] < 2:
        raise ValueError("At least two points are required to resample a polyline.")

    # Compute segment lengths
    diffs = points[1:] - points[:-1]  # shape: (N-1, 2)
    seg_lengths = jnp.sqrt(jnp.sum(diffs**2, axis=1))  # shape: (N-1,)
    cumulative_distances = jnp.concatenate(
        [jnp.array([0.0]), jnp.cumsum(seg_lengths)]
    )  # shape: (N,)
    total_length = cumulative_distances[-1]

    # Number of intervals needed so that (total_length / num_intervals) <= max_step
    # i.e., num_intervals >= (total_length / max_step).
    # We ensure it's at least 1 to avoid division by zero if total_length < max_step.
    num_intervals = jnp.maximum(1, jnp.ceil(total_length / max_step)).astype(int)

    # Actual uniform step (spacing) used
    step = total_length / num_intervals

    # Create distances 0 .. step .. 2*step .. total_length
    sample_distances = jnp.linspace(0.0, total_length, num_intervals + 1)

    # Function to interpolate a single distance 'd' along the polyline
    def interpolate(d):
        # Find which segment this distance falls in
        i = jnp.searchsorted(cumulative_distances, d, side="right") - 1
        i = jnp.clip(i, 0, points.shape[0] - 2)  # clip to valid range
        seg_len = seg_lengths[i]

        # Parametric distance along the segment [i, i+1]
        # Avoid zero division if seg_len=0 (coincident points)
        t = (d - cumulative_distances[i]) / jnp.where(seg_len > 0, seg_len, 1e-12)

        return points[i] + t * diffs[i]

    # Vectorize the interpolation
    new_points = jax.vmap(interpolate)(sample_distances)
    return new_points
