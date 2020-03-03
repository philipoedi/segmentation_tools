# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:25:05 2020

@author: OediP
"""

# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import segmentation_tools

x = np.arange(0,20,0.1)
sin_curve = np.sin(x)
saw_curve = sawtooth(x,0.1)
saw_sin = sin_curve*saw_curve 

#max_error = 1
max_error = 0.5
k = segmentation_tools.top_down()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error,"linear_regression")

k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")


k = segmentation_tools.top_down()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error,"linear_interpolation")

k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")



#step function
step = np.zeros(len(x))
size = int(len(step) / 4)
step[:size] = step[:size]+1
step[2*size:3*size] = step[2*size:3*size]+1
plt.plot(step)


k = segmentation_tools.top_down()

print("check step function segmentation with linear regression")
k.fit(step.reshape(len(saw_sin),1),max_error,"linear_regression")
k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")

print("check step function segmentation with linear interpolation")
k = segmentation_tools.top_down()
k.fit(step.reshape(len(saw_sin),1),max_error,"linear_interpolation")
k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")
