import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
import pandas as pd


import csv
import numpy as np
import matplotlib.pyplot as plt

data_path = './mean_scores.data'
plt.rcParams.update({'font.size': 18})

with open(data_path, 'r') as f:
    reader = csv.reader(f, delimiter='\t')
    # get header from first row
    headers = next(reader)
    # get all the rows as a list
    data = list(reader)
    # transform data into numpy array
    data = np.array(data).astype(float)

print(headers)
print(data.shape)

# Plot the data
fig, ax = plt.subplots()

ax.errorbar(data[:, 0], data[:, 1],
            yerr=data[:, 4],
            fmt='-o', label='Tabula Rasa')

ax.errorbar(data[:, 0], data[:, 2],
            yerr=data[:, 5],
            fmt='-*', label='Expert Bootstrapping Learning')

ax.errorbar(data[:, 0], data[:, 3],
            yerr=data[:, 6],
            fmt='-v', label='Exploratory Imitation')

plt.legend(loc='lower right')
ax.set_xlabel('Iteration')
ax.set_ylabel('Beagle #Steps / TRAIL #Steps')

plt.show()


