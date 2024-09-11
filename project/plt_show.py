import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example 3D entropy data, replace this with your actual data
entropy_data = np.random.rand(15, 15, 4)

# Get the dimensions of your data
dim_x, dim_y, dim_z = entropy_data.shape

# Generating normally distributed random data and adding to entropy_data
mean = 1
std = 1
random_3d_array = np.random.normal(mean, std, (dim_x, dim_y, dim_z))
entropy_data += random_3d_array

# Normalize entropy data
entropy_data_map = entropy_data / np.max(entropy_data)

# Generate voxel positions
xpos, ypos, zpos = np.meshgrid(np.arange(dim_x),
                               np.arange(dim_y),
                               np.arange(dim_z),
                               indexing="ij")

xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = zpos.ravel()

# Set the size of each voxel
dx = dy = dz = 1

# Prepare the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((np.ptp(xpos), np.ptp(ypos), np.ptp(zpos)))

# Set transparency (alpha) value
alpha = 0.5

# Initialize colors array
colors = np.zeros((len(entropy_data_map.ravel()), 4))

# Normalize the entropy_data_map to [0, 1]
norm = plt.Normalize(entropy_data_map.min(), entropy_data_map.max())
normalized = norm(entropy_data_map.ravel())

# Transition from green (0,1,0) to blue (0,0,1) based on the value
colors[:, 0] = 0  # Red channel is always 0
colors[:, 1] = 1 - normalized  # Green channel decreases with value
colors[:, 2] = normalized  # Blue channel increases with value
colors[:, 3] = alpha  # Alpha channel

# Draw the voxels
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, alpha=0.1)

# Add a color bar which maps values to colors, using a blue-green color map
mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
mappable.set_array(entropy_data_map)
fig.colorbar(mappable)

# Set axis labels
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# Set the view angle to better display the 3D effect
ax.view_init(elev=30, azim=30)

# Show the plot
plt.show()
