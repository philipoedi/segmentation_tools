# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import sawtooth
import matplotlib.pyplot as plt
import segmentation_tools

x = np.arange(0,20,0.1)
sin_curve = np.sin(x)
saw_curve = sawtooth(x,0.1)
saw_sin = sin_curve*saw_curve 

max_error = 0.7
k = segmentation_tools.sliding_window()
k.fit(saw_sin.reshape(len(saw_sin),1),max_error,"linear_interpolation")
k.segment_plot()
assert k.error > max_error, "error to big"
assert np.max(k.labels) == 6, "wrong segments"
print("everything fine")


step = np.zeros(len(x))
size = int(len(step) / 4)
step[:size] = step[:size]+1
step[2*size:3*size] = step[2*size:3*size]+1
plt.plot(step)


#step function
k = segmentation_tools.sliding_window()

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