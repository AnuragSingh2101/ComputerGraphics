import numpy as np
import matplotlib.pyplot as plt


def hybrid_transform(
    translation=(0, 0, 0),
    scaling=(1, 1, 1),
    rotation_angles=(0, 0, 0),  # radians (Rx, Ry, Rz)
    reflection_planes=(False, False, False),  # reflect about (X, Y, Z)
    shearing=(0, 0, 0, 0, 0, 0)  # sh_xy, sh_xz, sh_yx, sh_yz, sh_zx, sh_zy
):
    tx, ty, tz = translation
    sx, sy, sz = scaling
    rx, ry, rz = rotation_angles
    reflect_x, reflect_y, reflect_z = reflection_planes
    sh_xy, sh_xz, sh_yx, sh_yz, sh_zx, sh_zy = shearing

    # Translation matrix
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])

    # Scaling matrix
    S = np.array([
        [sx, 0, 0, 0],
        [0, sy, 0, 0],
        [0, 0, sz, 0],
        [0, 0, 0, 1]
    ])

    # Rotation matrices around X, Y, Z
    cx, sx_ = np.cos(rx), np.sin(rx)
    cy, sy_ = np.cos(ry), np.sin(ry)
    cz, sz_ = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1, 0, 0, 0],
        [0, cx, -sx_, 0],
        [0, sx_, cx, 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [cy, 0, sy_, 0],
        [0, 1, 0, 0],
        [-sy_, 0, cy, 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [cz, -sz_, 0, 0],
        [sz_, cz, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    R = Rz @ Ry @ Rx  # Combined rotation

    # Reflection matrix
    Rx_reflect = -1 if reflect_x else 1
    Ry_reflect = -1 if reflect_y else 1
    Rz_reflect = -1 if reflect_z else 1

    M = np.array([
        [Rx_reflect, 0, 0, 0],
        [0, Ry_reflect, 0, 0],
        [0, 0, Rz_reflect, 0],
        [0, 0, 0, 1]
    ])

    # Shearing matrix
    H = np.array([
        [1, sh_xy, sh_xz, 0],
        [sh_yx, 1, sh_yz, 0],
        [sh_zx, sh_zy, 1, 0],
        [0, 0, 0, 1]
    ])

    # Combine all transformations: T * H * M * R * S
    return T @ H @ M @ R @ S


def plot_cube(ax, verts, color='blue', label=None):
    x, y, z = verts[0], verts[1], verts[2]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
    ]

    ax.scatter(x, y, z, color=color)
    for e in edges:
        xs = [x[e[0]], x[e[1]]]
        ys = [y[e[0]], y[e[1]]]
        zs = [z[e[0]], z[e[1]]]
        ax.plot(xs, ys, zs, color=color)

    if label:
        ax.plot([], [], [], color=color, label=label)  # dummy plot for legend


def get_float(prompt, default=0.0):
    try:
        val = input(prompt)
        if val.strip() == '':
            print(f"Using default: {default}")
            return default
        return float(val)
    except ValueError:
        print(f"Invalid input. Using default value {default}")
        return default


def get_bool(prompt):
    ans = input(prompt + " (y/n): ").strip().lower()
    return ans == 'y'


if __name__ == "__main__":
    # Hardcoded cube vertices centered at origin (edges length=1)
    cube_vertices = np.array([
        [-0.5, -0.5, -0.5, 1],
        [ 0.5, -0.5, -0.5, 1],
        [ 0.5,  0.5, -0.5, 1],
        [-0.5,  0.5, -0.5, 1],
        [-0.5, -0.5,  0.5, 1],
        [ 0.5, -0.5,  0.5, 1],
        [ 0.5,  0.5,  0.5, 1],
        [-0.5,  0.5,  0.5, 1],
    ], dtype=float).T

    print("\nCube vertices (centered at origin):")
    print(cube_vertices[:3, :].T)

    print("\nEnter transformation parameters (leave blank for defaults):")

    # Translation
    do_translate = get_bool("Apply translation?")
    if do_translate:
        tx = get_float("Translate X: ")
        ty = get_float("Translate Y: ")
        tz = get_float("Translate Z: ")
    else:
        tx = ty = tz = 0

    # Scaling
    do_scale = get_bool("Apply scaling?")
    if do_scale:
        sx = get_float("Scale X: ", 1)
        sy = get_float("Scale Y: ", 1)
        sz = get_float("Scale Z: ", 1)
    else:
        sx = sy = sz = 1

    # Rotation
    do_rotate = get_bool("Apply rotation? (angles in degrees)")
    if do_rotate:
        rx = np.radians(get_float("Rotate around X (degrees): "))
        ry = np.radians(get_float("Rotate around Y (degrees): "))
        rz = np.radians(get_float("Rotate around Z (degrees): "))
    else:
        rx = ry = rz = 0

    # Reflection
    do_reflect = get_bool("Apply reflection?")
    if do_reflect:
        reflect_x = get_bool("Reflect about X-axis?")
        reflect_y = get_bool("Reflect about Y-axis?")
        reflect_z = get_bool("Reflect about Z-axis?")
    else:
        reflect_x = reflect_y = reflect_z = False

    # Shearing
    do_shear = get_bool("Apply shearing?")
    if do_shear:
        sh_xy = get_float("Shear XY: ")
        sh_xz = get_float("Shear XZ: ")
        sh_yx = get_float("Shear YX: ")
        sh_yz = get_float("Shear YZ: ")
        sh_zx = get_float("Shear ZX: ")
        sh_zy = get_float("Shear ZY: ")
    else:
        sh_xy = sh_xz = sh_yx = sh_yz = sh_zx = sh_zy = 0

    # Compute transformation matrix
    T_final = hybrid_transform(
        translation=(tx, ty, tz),
        scaling=(sx, sy, sz),
        rotation_angles=(rx, ry, rz),
        reflection_planes=(reflect_x, reflect_y, reflect_z),
        shearing=(sh_xy, sh_xz, sh_yx, sh_yz, sh_zx, sh_zy)
    )

    # Apply transformation
    transformed_vertices = T_final @ cube_vertices
    transformed_vertices /= transformed_vertices[3, :]  # normalize homogeneous

    # Combine original and transformed points to set plot limits
    all_points = np.hstack((cube_vertices[:3, :], transformed_vertices[:3, :]))
    mins = np.min(all_points, axis=1)
    maxs = np.max(all_points, axis=1)
    margin = 0.5
    limits = [(mins[i] - margin, maxs[i] + margin) for i in range(3)]

    # Plotting
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title("Original Cube (Centered at Origin)")
    plot_cube(ax1, cube_vertices, color='green', label='Original')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title("Transformed Cube")
    plot_cube(ax2, transformed_vertices, color='red', label='Transformed')

    for ax in (ax1, ax2):
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])
        ax.set_box_aspect([1, 1, 1])  # equal aspect ratio
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
