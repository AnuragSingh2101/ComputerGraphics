import matplotlib.pyplot as plt

# ----------------------------
# DDA Line Drawing Algorithm
# ----------------------------
def dda_line(x1, y1, x2, y2):
    points = []

    dx = x2 - x1
    dy = y2 - y1

    steps = int(max(abs(dx), abs(dy)))

    x_inc = dx / steps
    y_inc = dy / steps

    x, y = x1, y1
    for _ in range(steps + 1):
        points.append((round(x), round(y)))
        x += x_inc
        y += y_inc

    return points

# ----------------------------
# Bresenham's Line Drawing Algorithm
# ----------------------------
def bresenham_line(x1, y1, x2, y2):
    points = []

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    x, y = x1, y1
    sx = 1 if x2 > x1 else -1
    sy = 1 if y2 > y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x2:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        points.append((x, y))  # last point
    else:
        err = dy / 2.0
        while y != y2:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        points.append((x, y))  # last point

    return points

# ----------------------------
# Visualization Function
# ----------------------------
def plot_lines(dda_pts, bres_pts, x1, y1, x2, y2):
    dda_x, dda_y = zip(*dda_pts)
    bres_x, bres_y = zip(*bres_pts)

    plt.figure(figsize=(8, 8))
    plt.plot(dda_x, dda_y, 'ro-', label="DDA", linewidth=1)
    plt.plot(bres_x, bres_y, 'bs--', label="Bresenham", linewidth=1)
    plt.scatter([x1, x2], [y1, y2], color='green', zorder=5, label='Endpoints')

    plt.grid(True)
    plt.legend()
    plt.title("Line Drawing Algorithms: DDA vs Bresenham")
    plt.gca().invert_yaxis()  # Invert y-axis to mimic screen coordinate system
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

# ----------------------------
# Main Function
# ----------------------------
def main():
    print("Enter two points (x1, y1) and (x2, y2):")
    x1 = int(input("x1: "))
    y1 = int(input("y1: "))
    x2 = int(input("x2: "))
    y2 = int(input("y2: "))

    # Generate points using both algorithms
    dda_points = dda_line(x1, y1, x2, y2)
    bres_points = bresenham_line(x1, y1, x2, y2)

    # Print points
    print("\nDDA Line Points:")
    print(dda_points)

    print("\nBresenham Line Points:")
    print(bres_points)

    # Plot the results
    plot_lines(dda_points, bres_points, x1, y1, x2, y2)

# ----------------------------
# Run Program
# ----------------------------
if __name__ == "__main__":
    main()
