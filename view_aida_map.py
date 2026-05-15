#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    map_file = os.path.expanduser('~/.aida/aida_memory_map.json')
    if not os.path.exists(map_file):
        print(f"Map file not found: {map_file}")
        return

    with open(map_file, 'r') as f:
        data = json.load(f)

    # Track bounds
    X_MAX = 1.184
    Y_MAX = 0.781

    # Map arrays for scatter
    hazard_x = []
    hazard_y = []
    track_x = []
    track_y = []

    for key, val in data.items():
        try:
            x_str, y_str = key.split('_')
            x = float(x_str)
            y = float(y_str)
        except ValueError:
            continue

        if not (0.0 <= x <= X_MAX and 0.0 <= y <= Y_MAX):
            continue

        confidence = val.get('confidence', 0.0)

        if confidence >= 0.80:
            hazard_x.append(x)
            hazard_y.append(y)
        else:
            track_x.append(x)
            track_y.append(y)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot track boundaries
    if track_x:
        ax.scatter(track_x, track_y, c='black', marker='s', s=10, label='Track (<0.80 conf)')

    # Plot hazards
    if hazard_x:
        ax.scatter(hazard_x, hazard_y, c='red', marker='s', s=10, label='Hazard (>=0.80 conf)')

    # Set display bounds exactly to track limits
    ax.set_xlim(0.0, X_MAX)
    ax.set_ylim(0.0, Y_MAX)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('AIDA Spatial Memory Map')

    if track_x or hazard_x:
        ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.show()

if __name__ == "__main__":
    main()
