import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D

# /media/ye/Data/AnyDrive_GT/S_LS/eccv_map.pkl

# 假设 entropy_data 是你的三维熵数据数组
# 这里我们创建一个具有随机熵值的示例数组
entropy_data_map = np.random.rand(10, 10, 4)  # 使用随机数据作为示例

# with open('/media/ye/Data/AnyDrive_GT/S_LS/eccv_map.pkl', 'rb') as file:
#     entropy_data_map = pickle.load(file)
# 获取数据的尺寸
dim_x, dim_y, dim_z = entropy_data_map.shape

# 为了绘制体素，我们需要知道每个体素的角落位置
xpos, ypos, zpos = np.meshgrid(np.arange(dim_x),
                               np.arange(dim_y),
                               np.arange(dim_z),
                               indexing="ij")

xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = zpos.ravel()

# 设置体素的大小
dx = dy = dz = 1

# 设置透明度，这里设为 0.5，你可以根据需要调整这个值
alpha = 0.5

# 准备绘制图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用体素的熵值作为颜色映射，并加入透明度
colors = plt.cm.jet(entropy_data_map.ravel() / entropy_data_map.max())
# 更新颜色数组以包含透明度信息
# colors = np.insert(colors, 3, alpha, axis=1)

# 绘制体素
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, alpha=0.1)

# 添加颜色条
mappable = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=entropy_data_map.min(), vmax=entropy_data_map.max()))
mappable.set_array(entropy_data_map)
fig.colorbar(mappable)

# 设置轴标签
ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

# 设置视图角度以更好地显示三维效果
ax.view_init(elev=30, azim=30)

# 显示图像
plt.show()
