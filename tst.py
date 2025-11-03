import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Example scatter data with different scales
x = np.random.rand(100)
y = 100 * np.random.rand(100)  # Y has different scale

fig, ax = plt.subplots(figsize=(6, 2))  # Wide figure
ax.scatter(x, y)

# Define triangle in axes coordinates (0 = left/bottom, 1 = right/top)
triangle = patches.Polygon(
    [[0, 0], [0.2, 0], [0, 0.2]],   # Triangle points: bottom-left, right, top
    closed=True,
    transform=ax.transAxes,         # Interpret in axes coordinates!
    color='lightblue',
    alpha=0.4,
    clip_on=False
)

ax.add_patch(triangle)

# Set data limits as usual
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)

plt.show()
