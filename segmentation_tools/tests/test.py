# -*- coding: utf-8 -*-
import numpy as np
from scipy.signal import sawtooth

import segmentation_tools

x = np.arange(0,20,0.1)
sin_curve = np.sin(x)
saw_curve = sawtooth(x,0.1)
saw_sin = sin_curve*saw_curve 

k = segmentation_tools.sliding_window1()
k.fit(saw_sin,1)

