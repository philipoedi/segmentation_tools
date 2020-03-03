# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:25:19 2020

@author: OediP
"""

import numpy as np
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import segmentation_tools

x = np.arange(0,20,0.1)
sin_curve = np.sin(x)
saw_curve = sawtooth(x,0.1)
saw_sin = sin_curve*saw_curve 

#linear regression
max_error = 0.5
print("check saw sin function segmentation with linear regression")
k = segmentation_tools.bottom_up()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error,"linear_regression")
k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")


#linear interpolation
max_error= 0.5
print("check saw sin function segmentation with linear interpolation")
k = segmentation_tools.bottom_up()
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


print("check step function segmentation with linear regression")
k = segmentation_tools.bottom_up()
k.fit(step.reshape(len(saw_sin),1),max_error,"linear_regression")

# check number of labels and segment borders
k.segment_plot()
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")

print("check step function segmentation with linear interpolation")
k = segmentation_tools.bottom_up()
k.fit(step.reshape(len(saw_sin),1),max_error,"linear_interpolation")
# check number of labels and segment borders
k.segment_plot()
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")

