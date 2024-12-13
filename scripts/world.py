from microcity.terrain import make_terrain_mesh

import numpy as np
import pygfx as gfx
from wgpu.gui.auto import WgpuCanvas, run


GRID_SCALE = 5.0
HEIGHT_SCALE = 0.2 * GRID_SCALE


def make_in_progress_road_geometry(waypoints: np.array) -> gfx.WorldObject:
    color = (0, 1, 1, 1)

    print(f"waypoints: {waypoints}")
    positions = waypoints.copy()
    positions[:, 2] += HEIGHT_SCALE
    geometry = gfx.Geometry(positions=positions)
    material = gfx.PointsMaterial(color=color, size=20)
    points = gfx.Points(geometry, material)

    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(color=color, thickness=0.1 * GRID_SCALE),
    )

    g = gfx.WorldObject()
    g.add(points)
    g.add(line)

    return g


def get_pick_coords(geom, info):
    face_index = info["face_index"]
    face_coord = info["face_coord"]
    vert_indices = geom.indices.data[face_index]
    # TODO EDF use bary coords
    return geom.positions.data[vert_indices[0]]


def main():
    terrain = make_terrain_mesh(GRID_SCALE)

    # Visualize the mesh using pygfx
    canvas = WgpuCanvas()
    renderer = gfx.renderers.WgpuRenderer(canvas)

    scene = gfx.Scene()
    scene.add(terrain)

    dark_gray = np.array((169, 167, 168, 255)) / 255
    light_gray = np.array((100, 100, 100, 255)) / 255
    background = gfx.Background.from_color(light_gray, dark_gray)
    scene.add(background)

    scene.add(gfx.AmbientLight())
    scene.add(gfx.DirectionalLight())

    camera = gfx.PerspectiveCamera(70, 16 / 9)
    camera.local.z = 150.0
    camera.show_object(scene)
    # controller = gfx.OrbitController(camera, register_events=renderer)
    controller = gfx.PanZoomController(camera, register_events=renderer)

    click_waypoints = None
    hover_point = None

    def terrain_click(event: gfx.Event):
        nonlocal click_waypoints
        xyz = get_pick_coords(terrain.geometry, event.pick_info)
        if click_waypoints is None:
            click_waypoints = []
        click_waypoints.append(xyz)

    def terrain_hover(event: gfx.Event):
        if click_waypoints is None:
            return
        xyz = get_pick_coords(terrain.geometry, event.pick_info)
        hover_point = xyz
        # TODO EDF draw provisional waypoint

    def terrain_double_click(event: gfx.Event):
        nonlocal click_waypoints
        click_waypoints = None
        # TODO EDF add road

    terrain.add_event_handler(terrain_click, "pointer_down")
    terrain.add_event_handler(terrain_double_click, "double_click")
    terrain.add_event_handler(terrain_hover, "pointer_move")

    interaction_viz: gfx.WorldObject = gfx.WorldObject()
    scene.add(interaction_viz)

    def key_press(event: gfx.Event):
        nonlocal click_waypoints
        assert isinstance(event, gfx.KeyboardEvent)
        if event.key == 'Escape':
            click_waypoints = None
            canvas.request_draw()
        

    renderer.add_event_handler(key_press, "key_down")

    def add_interaction_viz(*args):
        if click_waypoints is None:
            return
        g = make_in_progress_road_geometry(np.array(click_waypoints, dtype=np.float32))
        interaction_viz.add(g)

    def remove_interaction_viz(*args):
        interaction_viz.clear()

    def frame():
        add_interaction_viz()
        renderer.render(scene, camera)
        remove_interaction_viz()

    canvas.request_draw(frame)

    run()


if __name__ == "__main__":
    main()
