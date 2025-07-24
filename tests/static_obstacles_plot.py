#! /usr/bin/python3

import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_bounding_boxes(bounding_boxes, limit_count):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    i = 0
    # Plot each bounding box
    for bbox in bounding_boxes:
        i = i + 1
        if i == limit_count:
            break

        # Extract bounding box limits
        x_min, x_max = bbox["x_min"], bbox["x_max"]
        y_min, y_max = bbox["y_min"], bbox["y_max"]
        z_min, z_max = bbox["z_min"], bbox["z_max"]

        # Define the 8 vertices of the bounding box
        vertices = [
            [x_min, y_min, z_min],
            [x_min, y_max, z_min],
            [x_max, y_max, z_min],
            [x_max, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_max],
            [x_max, y_max, z_max],
            [x_max, y_min, z_max]
        ]

        # Define the 12 edges (pairs of vertices)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        # Plot edges
        for edge in edges:
            points = [vertices[edge[0]], vertices[edge[1]]]
            x, y, z = zip(*points)
            ax.plot(x, y, z, color="b")

        # Add the bounding box as a translucent surface
        faces = [
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # -X face
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # +X face
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # -Y face
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # +Y face
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom face
            [vertices[4], vertices[5], vertices[6], vertices[7]]   # Top face
        ]
        ax.add_collection3d(Poly3DCollection(faces, alpha=0.2, edgecolor="r"))

    # Set plot limits and labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Bounding Boxes in 3D Space")
    plt.show()

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot bounding boxes from a file.")
    parser.add_argument('--file', '-f', type=str, default='../utils/tmp/static_obstacles.json',
                        help="Path to the JSON file containing bounding box data.")
    parser.add_argument('--limit_count', '-c', default=100, help='Plot limited number of boxes.')
    args = parser.parse_args()

    # Read bounding boxes from file
    with open(args.file, "r") as file:
        bounding_boxes = json.load(file)
    print('Total number of bounding boxes in file =', len(bounding_boxes))
    # Validate the data
    if not isinstance(bounding_boxes, list):
        raise ValueError("The input file must contain a list of bounding box dictionaries.")

    # Plot the bounding boxes
    plot_bounding_boxes(bounding_boxes, args.limit_count)

if __name__ == "__main__":
    main()
