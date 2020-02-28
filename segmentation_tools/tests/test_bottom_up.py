# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:25:19 2020

@author: OediP
"""

import numpy as np
from scipy.signal import sawtooth

import segmentation_tools

x = np.arange(0,20,0.1)
sin_curve = np.sin(x)
saw_curve = sawtooth(x,0.1)
saw_sin = sin_curve*saw_curve 

#max_error = 1
max_error = 0.5
k = segmentation_tools.bottom_up()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error,"linear_regression")
print(k.segment_borders)
k.segment_plot()

# check number of labels and segment borders
np.max(k.labels)+1 == len(k.segment_borders)

# check max_error
np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error

print("everything fine")



#max_error = 1
max_error = 0.5
k = segmentation_tools.bottom_up()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error)
print(k.segment_borders)
k.segment_plot()

# check number of labels and segment borders
np.max(k.labels)+1 == len(k.segment_borders)

# check max_error
np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error

print("everything fine")








step = np.zeros(len(x))
size = int(len(step) / 4)
step[:size] = step[:size]+1
step[2*size:3*size] = step[2*size:3*size]+1
plt.plot(step)


#step function
k = segmentation_tools.bottom_up()

print("check step function segmentation with linear regression")
k.fit(step.reshape(len(saw_sin),1),max_error,"linear_regression")
assert k.error < max_error, "error to big"
assert np.max(k.labels) == 3, "wrong segments"
print("everything fine")

print("check step function segmentation with linear interpolation")
k.fit(step.reshape(len(saw_sin),1),max_error,"linear_interpolation")
assert k.error < max_error, "error to big"
assert np.max(k.labels) == 3, "wrong segments"
print("everything fine")

plt.plot(k.labels)
plt.plot(step)
plt.show()

test_list = [[1,3],[23,3,3],[1]]
np.concatenate([np.ones(len(test_list[i]))*i for i in range(len(test_list))])