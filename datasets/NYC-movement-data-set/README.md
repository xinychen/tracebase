
- **Draw speed scatters**

```python
import numpy as np
import matplotlib.pyplot as plt

dense_mat = np.load('../datasets/NYC-movement-data-set/hourly_speed_mat_2019_1.npz')['arr_0']
for month in range(2, 4):
    dense_mat = np.append(dense_mat, np.load('../datasets/NYC-movement-data-set/hourly_speed_mat_2019_{}.npz'.format(month))['arr_0'], axis = 1)

plt.figure(figsize = (7.5, 3.5))
ax1 = plt.subplot(2, 3, 1)
data = dense_mat[0, : 24]
pos1 = np.where(data > 0)
pos2 = np.where(data == 0)
plt.plot(pos1[0], data[pos1], 'bo', markerfacecolor = 'white')
plt.plot(pos2[0], data[pos2], 'o', markerfacecolor = 'white', markeredgecolor = 'black')
ax1.tick_params(direction = "in")
plt.ylim([-2, 30])
plt.ylabel('Speed (mph)')

ax2 = plt.subplot(2, 3, 2)
data = dense_mat[1, : 24]
pos1 = np.where(data > 0)
pos2 = np.where(data == 0)
plt.plot(pos1[0], data[pos1], 'bo', markerfacecolor = 'white')
plt.plot(pos2[0], data[pos2], 'o', markerfacecolor = 'white', markeredgecolor = 'black')
ax2.tick_params(direction = "in")
plt.ylim([-2, 30])

ax3 = plt.subplot(2, 3, 3)
data = dense_mat[2, : 24]
pos1 = np.where(data > 0)
pos2 = np.where(data == 0)
plt.plot(pos1[0], data[pos1], 'bo', markerfacecolor = 'white')
plt.plot(pos2[0], data[pos2], 'o', markerfacecolor = 'white', markeredgecolor = 'black')
ax3.tick_params(direction = "in")
plt.ylim([-2, 30])

ax4 = plt.subplot(2, 3, 4)
data = dense_mat[3, : 24]
pos1 = np.where(data > 0)
pos2 = np.where(data == 0)
plt.plot(pos1[0], data[pos1], 'bo', markerfacecolor = 'white')
plt.plot(pos2[0], data[pos2], 'o', markerfacecolor = 'white', markeredgecolor = 'black')
ax4.tick_params(direction = "in")
plt.ylim([-2, 30])
plt.xlabel('Time step (hour)')
plt.ylabel('Speed (mph)')

ax5 = plt.subplot(2, 3, 5)
data = dense_mat[4, : 24]
pos1 = np.where(data > 0)
pos2 = np.where(data == 0)
plt.plot(pos1[0], data[pos1], 'bo', markerfacecolor = 'white')
plt.plot(pos2[0], data[pos2], 'o', markerfacecolor = 'white', markeredgecolor = 'black')
ax5.tick_params(direction = "in")
plt.ylim([-2, 30])
plt.xlabel('Time step (hour)')

ax6 = plt.subplot(2, 3, 6)
data = dense_mat[5, : 24]
pos1 = np.where(data > 0)
pos2 = np.where(data == 0)
plt.plot(pos1[0], data[pos1], 'bo', markerfacecolor = 'white')
plt.plot(pos2[0], data[pos2], 'o', markerfacecolor = 'white', markeredgecolor = 'black')
ax6.tick_params(direction = "in")
plt.ylim([-2, 30])
plt.xlabel('Time step (hour)')

plt.savefig('NYC_missing_data_scatter.pdf', bbox_inches = 'tight')
plt.show()
```
