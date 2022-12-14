import numpy as np
import matplotlib.pyplot as plt
# if using a Jupyter notebook, include:
import pandas as pd


import csv
import numpy as np
import matplotlib.pyplot as plt

data_path = './completion.data'
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
# plt.rcParams.update({'font.size': 16})
# Plot the data
fig, ax = plt.subplots()

ax.errorbar(data[:, 0], 100*data[:, 1],
            yerr=100*data[:, 4],
            fmt='-o', label='Tabula Rasa')

ax.errorbar(data[:, 0], 100*data[:, 2],
            yerr=100*data[:, 5],
            fmt='-*', label='Expert Bootstrapping Learning')

ax.errorbar(data[:, 0], 100*data[:, 3],
            yerr=100*data[:, 6],
            fmt='-v', label='Exploratory Imitation')

plt.legend(loc='lower right')
ax.set_xlabel('Iteration')
ax.set_ylabel('Percentage of Problems Solved')
# ax.set_title('Line plot with error bars')
plt.show()
# plt.savefig('/Users/ibrahimabdelaziz/Documents/completition_ratios_new.png')
# plt.plot(data[:, 2])
# plt.xlabel('Table Index')
# plt.ylabel(headers[2])
# plt.show()
#
#
# x = np.linspace(0,5.5,10)
# y = 10*np.exp(-x)
# xerr = np.random.random_sample((10))
# yerr = np.random.random_sample((10))
#
# fig, ax = plt.subplots()
#
# ax.errorbar(x, y,
#             xerr=xerr,
#             yerr=yerr,
#             fmt='-o')
#
# ax.set_xlabel('x-axis')
# ax.set_ylabel('y-axis')
# ax.set_title('Line plot with error bars')
#
# plt.show()
