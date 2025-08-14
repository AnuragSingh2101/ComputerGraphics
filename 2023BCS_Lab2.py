"""Interactive 2D Transformations Demo with Triangle"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math


def setup_plot(vertices):
    """Setup and return plot axes with auto-limits based on vertices"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Find bounds of shape
    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])

    
    pad_x = (max_x - min_x) * 0.5 if max_x != min_x else 50
    pad_y = (max_y - min_y) * 0.5 if max_y != min_y else 50

    ax.set_xlim(min_x - pad_x, max_x + pad_x)
    ax.set_ylim(min_y - pad_y, max_y + pad_y)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Force grid spacing = 1 unit
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    return ax


def draw_shape(ax, vertices, color='blue'):
    """Draw polygon from vertices"""
    x = list(vertices[:, 0]) + [vertices[0, 0]]
    y = list(vertices[:, 1]) + [vertices[0, 1]]
    ax.plot(x, y, color=color, linewidth=2)

def apply_matrix(vertices, matrix):
    """Apply transformation matrix to vertices"""
    homogeneous = np.ones((len(vertices), 3))
    homogeneous[:, :2] = vertices
    return (matrix @ homogeneous.T).T[:, :2]

def translate(vertices, tx, ty):
    matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return apply_matrix(vertices, matrix)

def scale(vertices, sx, sy, pivot_x=0, pivot_y=0):
    t1 = np.array([[1, 0, -pivot_x], [0, 1, -pivot_y], [0, 0, 1]])
    s = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])
    t2 = np.array([[1, 0, pivot_x], [0, 1, pivot_y], [0, 0, 1]])
    return apply_matrix(vertices, t2 @ s @ t1)

def rotate(vertices, angle_deg, pivot_x=0, pivot_y=0):
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    t1 = np.array([[1, 0, -pivot_x], [0, 1, -pivot_y], [0, 0, 1]])
    r = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    t2 = np.array([[1, 0, pivot_x], [0, 1, pivot_y], [0, 0, 1]])
    return apply_matrix(vertices, t2 @ r @ t1)

def reflect_x(vertices):
    matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    return apply_matrix(vertices, matrix)

def shear_x(vertices, factor):
    matrix = np.array([[1, factor, 0], [0, 1, 0], [0, 0, 1]])
    return apply_matrix(vertices, matrix)

def show_transformation(name, original_vertices, transformed_vertices, description):
    combined = np.vstack((original_vertices, transformed_vertices))
    ax = setup_plot(combined)  # Auto zoom for both original and transformed
    draw_shape(ax, original_vertices, 'blue')
    draw_shape(ax, transformed_vertices, 'red')
    ax.set_title(f'{name}\n{description}', fontsize=14, fontweight='bold')
    ax.legend(['Original', 'Transformed'], loc='upper right')
    plt.show()
    plt.close()


def main():
    print("Enter triangle vertices (x y) format:")
    vertices = []
    for i in range(3):
        x, y = map(float, input(f"Vertex {i+1}: ").split())
        vertices.append([x, y])
    vertices = np.array(vertices)

    while True:
        print("\nChoose transformation:")
        print("1. Translation")
        print("2. Scaling")
        print("3. Rotation")
        print("4. Reflection across X-axis")
        print("5. Shearing along X-axis")
        print("6. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            tx, ty = map(float, input("Enter translation (tx ty): ").split())
            transformed = translate(vertices, tx, ty)
            show_transformation("Translation", vertices, transformed, f"Moved by ({tx}, {ty})")

        elif choice == "2":
            sx, sy = map(float, input("Enter scale factors (sx sy): ").split())
            px, py = map(float, input("Enter pivot point (px py): ").split())
            transformed = scale(vertices, sx, sy, px, py)
            show_transformation("Scaling", vertices, transformed, f"Scaled by ({sx}, {sy}) about ({px}, {py})")

        elif choice == "3":
            angle = float(input("Enter rotation angle in degrees: "))
            px, py = map(float, input("Enter pivot point (px py): ").split())
            transformed = rotate(vertices, angle, px, py)
            show_transformation("Rotation", vertices, transformed, f"Rotated by {angle}° about ({px}, {py})")

        elif choice == "4":
            transformed = reflect_x(vertices)
            show_transformation("Reflection", vertices, transformed, "Reflected across X-axis")

        elif choice == "5":
            factor = float(input("Enter shear factor: "))
            transformed = shear_x(vertices, factor)
            show_transformation("Shearing", vertices, transformed, f"Sheared along X-axis with factor {factor}")

        elif choice == "6":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
