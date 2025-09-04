import numpy as np
import matplotlib.pyplot as plt


cube_vertices = np.array([
    [-1, -1, -1],
    [1, -1, -1],
    [1,  1, -1],
    [-1, 1, -1],
    [-1, -1,  1],
    [1, -1,  1],
    [1,  1,  1],
    [-1, 1,  1]
])


cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  
    (4, 5), (5, 6), (6, 7), (7, 4),  
    (0, 4), (1, 5), (2, 6), (3, 7)   
]

orthogonal_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0]
])

def perspective_projection(point, d=3):
    """ Project 3D point into 2D using perspective projection """
    x, y, z = point
    factor = d / (z + d)  # scaling factor
    return np.array([x * factor, y * factor]) 

orthogonal_proj = np.array([orthogonal_matrix @ v for v in cube_vertices])

perspective_proj = np.array([perspective_projection(v) for v in cube_vertices])

# ---- Visualization ----
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].set_title("Orthogonal Projection")
for edge in cube_edges:
    p1, p2 = orthogonal_proj[edge[0]], orthogonal_proj[edge[1]]
    axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-')
axes[0].set_aspect('equal')

axes[1].set_title("Perspective Projection")
for edge in cube_edges:
    p1, p2 = perspective_proj[edge[0]], perspective_proj[edge[1]]
    axes[1].plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
axes[1].set_aspect('equal')

plt.show()

