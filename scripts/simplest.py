import pygfx as gfx


def main():
    cube = gfx.Mesh(
        gfx.box_geometry(200, 200, 200),
        gfx.MeshPhongMaterial(color="#336699"),
    )

    gfx.show(cube)


if __name__ == "__main__":
    main()
