There are some Python codes for visualizing results in this work.

- **The histogram of ground truth data and forecasts achieved by NoTMF with $\delta=6$ and $d=6$ in the test set of the NYC dataset. The missing rate implies the ratio of missing values of road segments in the test set.**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for k in range(1, 11):
    if k == 1:
        pos = np.where(np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) <= 0.1 * 7 * 24)
    elif k == 10:
        pos = np.where(np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) > 0.9 * 7 * 24)
    else:
        pos = np.where((np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) > 0.1 * (k - 1) * 7 * 24)
                       & (np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) <= 0.1 * k * 7 * 24))
    print('{} road segments in that missing range.'.format(len(pos[0])))
    mat1 = dense_mat[pos[0], - 7 * 24 :]
    mat2 = mat_hat[pos[0], :]
    pos_new = np.where(mat1 > 0)

    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize = (4, 2))
    ax = fig.add_subplot(1, 1, 1)

    sns.distplot(mat1[pos_new], hist = False, kde = True,
                  kde_kws = {'linewidth': 3, 'color': 'orange', 'alpha': 0.65},
                  label = 'Ground truth')
    sns.distplot(mat2[pos_new], hist = False, kde = True,
                  kde_kws = {'linewidth': 3, 'color': 'red', 'alpha': 0.65},
                  label = 'NoTMF')
    plt.tick_params(direction = "in")
    plt.xlim([0, 80])
    ax.legend(['Ground truth', 'NoTMF'])
    plt.xlabel('Speed (mph)')
    plt.ylabel('Probability')
    plt.savefig("notmf_forecasts_hist_{}.pdf".format(k), bbox_inches = "tight")
    plt.show()
````

- 

- **The histogram of ground truth data (`dense_mat`) and forecasts (`mat_hat`).**

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for k in range(1, 11):
    if k == 1:
        pos = np.where(np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) <= 0.1 * 7 * 24)
    elif k == 10:
        pos = np.where(np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) > 0.9 * 7 * 24)
    else:
        pos = np.where((np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) > 0.1 * (k - 1) * 7 * 24)
                       & (np.sum(dense_mat[:, - 7 * 24 :] == 0, axis = 1) <= 0.1 * k * 7 * 24))
    print('{} road segments in that missing range.'.format(len(pos[0])))
    mat1 = dense_mat[pos[0], - 7 * 24 :]
    mat2 = mat_hat[pos[0], :]
    pos_new = np.where(mat1 > 0)

    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize = (4, 2))
    ax = fig.add_subplot(1, 1, 1)

    sns.distplot(mat1[pos_new], hist = False, kde = True,
                  kde_kws = {'linewidth': 3, 'color': 'orange', 'alpha': 0.65},
                  label = 'Ground truth')
    sns.distplot(mat2[pos_new], hist = False, kde = True,
                  kde_kws = {'linewidth': 3, 'color': 'red', 'alpha': 0.65},
                  label = 'NoTMF')
    plt.tick_params(direction = "in")
    plt.xlim([0, 80])
    ax.legend(['Ground truth', 'NoTMF'])
    plt.xlabel('Speed (mph)')
    plt.ylabel('Probability')
    plt.savefig("notmf_forecasts_hist_{}.pdf".format(k), bbox_inches = "tight")
    plt.show()
```
