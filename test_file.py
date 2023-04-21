import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 1)
data = np.random.random((10, 3))
columns = ("Column I", "Column II", "Column III")
axs.axis('tight')
axs.axis('off')
the_table = axs.table(cellText=data, colLabels=columns, loc='center')
plt.show()