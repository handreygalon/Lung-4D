from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

'''
X = np.array([-90.905383, -90.905383, -90.905383, -81.061633, -81.061633, -81.061633, -71.217883, -71.217883, -71.217883, -61.374133, -61.374133, -61.374133, -51.530383, -51.530383, -51.530383, -40.046008, -40.046008, -40.046008, -30.202258, -30.202258, -30.202258, -20.358508, -20.358508, -20.358508])

Y = np.array([20., 30., 40., 20., 30., 40., 20., 30., 40., 20., 30., 40., 20., 30., 40., 20., 30., 40., 20., 30., 40., 20., 30., 40.])

Z = np.array([-72.916493, -72.916493, -76.197743, -69.635243, -71.275868, -72.916493, -66.353993, -67.994618, -71.275868, -64.713368, -66.353993, -69.635243, -64.713368, -66.353993, -67.994618, -66.353993, -69.635243, -71.275868, -69.635243, -71.275868, -76.197743, -72.916493, -76.197743, -82.760243])
'''

X = np.array([[-90.905383, -90.905383, -90.905383],
              [-81.061633, -81.061633, -81.061633],
              [-71.217883, -71.217883, -71.217883],
              [-61.374133, -61.374133, -61.374133],
              [-51.530383, -51.530383, -51.530383],
              [-40.046008, -40.046008, -40.046008],
              [-30.202258, -30.202258, -30.202258],
              [-20.358508, -20.358508, -20.358508]])
# X = X.flatten()

Y = np.array([[20., 30., 40.],
              [20., 30., 40.],
              [20., 30., 40.],
              [20., 30., 40.],
              [20., 30., 40.],
              [20., 30., 40.],
              [20., 30., 40.],
              [20., 30., 40.]])
# Y = Y.flatten()

Z = np.array([[-72.916493, -72.916493, -76.197743],
              [-69.635243, -71.275868, -72.916493],
              [-66.353993, -67.994618, -71.275868],
              [-64.713368, -66.353993, -69.635243],
              [-64.713368, -66.353993, -67.994618],
              [-66.353993, -69.635243, -71.275868],
              [-69.635243, -71.275868, -76.197743],
              [-72.916493, -76.197743, -82.760243]])
# Z = Z.flatten()

fig = plt.figure()
ax = fig.gca(projection='3d')
# ax.plot_trisurf(X, Y, Z, linewidth=0.2, antialiased=True)
ax.plot_wireframe(X, Y, Z)

# Create cubic bounding box to simulate equal aspect ratio
max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

# Comment or uncomment following both lines to test the fake bounding box:
for xb, yb, zb in zip(Xb, Yb, Zb):
    ax.plot([xb], [yb], [zb], 'w')

plt.grid()

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
