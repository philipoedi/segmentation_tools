# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:25:31 2020

@author: OediP
"""

import numpy as np
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import segmentation_tools

x = np.arange(0,50,0.1)
sin_curve = np.sin(x)
saw_curve = sawtooth(x,0.1)
saw_sin = sin_curve*saw_curve 

max_error = 0.5
k = segmentation_tools.SWAB()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error,plr="linear_interpolation",error_type = "max",buffer_size = 100)

k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")




saw_sin_2d = np.array([saw_sin,saw_sin])
max_error = 0.5
k = segmentation_tools.SWAB()
k.fit(saw_sin_2d.T,max_error,plr="linear_regression",error_type = "max",buffer_size = 100)

k.segment_plot()
# check number of labels and segment borders
assert np.max(k.labels)+1 == len(k.segment_borders),"num labels don't match"
# check max_error
assert np.max([k.calculate_error(k.segments[i].data) for i in range(len(k.segments))]) < max_error,"error to big"
print("everything fine")
