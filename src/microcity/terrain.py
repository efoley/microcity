import numpy as np
import pygfx as gfx

from scipy.ndimage import gaussian_filter

GEN = np.random.default_rng()


int2 = tuple[int, int]


def _make_random_heightmap(
    shape: int2,
    smoothing_radius: float = 10.0,
    height_scale: float = 20.0,
    gen: np.random.Generator | None = None,
):
    """
    Generate a dumb smoothed random normal heightmap.
    """
    if gen is None:
        gen = GEN
    heights = height_scale * gen.standard_normal(shape)
    smoothed_heights = gaussian_filter(heights, sigma=smoothing_radius)
    return smoothed_heights


def _make_flat_heightmap(
    shape: int2,
):
    return np.ones(shape)


def _create_mesh_from_heightmap(heightmap, scale=(1.0, 1.0, 1.0)):
    """
    Create a 3D mesh from a heightmap using pygfx.

    Parameters:
        heightmap (np.ndarray): The heightmap array.
        scale (tuple): Scaling factors for x, y, and z dimensions.

    Returns:
        gfx.Mesh: A pygfx mesh object representing the heightmap.
    """
    rows, cols = heightmap.shape

    # Create the vertex grid
    x = np.linspace(0, cols * scale[0], cols)
    y = np.linspace(0, rows * scale[1], rows)
    x, y = np.meshgrid(x, y)
    z = heightmap * scale[2]

    # Flatten the arrays for vertices
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)

    # Create the triangle indices
    indices = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            top_left = i * cols + j
            top_right = top_left + 1
            bottom_left = (i + 1) * cols + j
            bottom_right = bottom_left + 1

            # Two triangles per grid square
            indices.append([top_left, bottom_left, top_right])
            indices.append([top_right, bottom_left, bottom_right])

    indices = np.array(indices, dtype=np.uint32)

    # Create the mesh
    geometry = gfx.Geometry(positions=vertices, indices=indices)
    material = gfx.MeshPhongMaterial(color=(0.5, 0.8, 0.5), pick_write=True)
    mesh = gfx.Mesh(geometry, material)

    return mesh


def make_terrain_mesh(grid_scale: float):
    shape = (1024, 1024)
    heights = _make_flat_heightmap(shape)
    return _create_mesh_from_heightmap(heights, scale=(grid_scale, grid_scale, 1.0))
