import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example 3D entropy data, replace this with your actual data
entropy_data = np.random.rand(10, 10, 5)


# Get the dimensions of your data
dim_x, dim_y, dim_z = entropy_data.shape

mean = 1
std = 1

# 生成遵循正态分布的三维随机数数组
random_3d_array = np.random.normal(mean, std, (dim_x, dim_y, dim_z))

entropy_data += random_3d_array

entropy_data_map = entropy_data / 136000

# Generate the voxel positions
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
# Set the transparency (alpha) value
alpha = 0.1

# Map the entropy values to a green color map with varying alpha based on the value
colors = np.zeros((len(entropy_data_map.ravel()), 4))
norm = plt.Normalize(entropy_data_map.min(), entropy_data_map.max())
colors[:, 1] = 0.8  # set green channel to 100%
# colors[:, 1] = colors[:, 1] * colors[:, 1]
colors[:, 3] = norm(entropy_data_map.ravel()) * alpha  # set alpha channel

# Draw the voxels
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True)

# Add a color bar which maps values to colors
mappable = plt.cm.ScalarMappable(cmap=plt.cm.Greens, norm=norm)
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
