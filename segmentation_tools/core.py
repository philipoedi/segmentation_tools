# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:41:27 2020

@author: OediP
"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

# to do:
# swab implementation
# add different error types: absolute error,squared
# remove class segmen
# indices instead of data
# develop better test cases
# for loop optimization


class segment():
    """
    """
    def __init__(self,data,error):
        self.data = data
    
        
class estimator:
    """
    """
    def __init__(self):
        self.max_error = None
        self.labels = None
        self.data = None
        self.algorithm = None
        self.error = 0
        self.plr = None
        self.segments = list()
        self.calculate_error = None
        self.segment_borders = list()
        self.labels = None
        self.error_type = None
        
    def fit(self,data,max_error,plr):
        self.labels = np.zeros(data.shape[0])
        self.max_error = max_error
        self.data = data
        self.plr = plr
        if plr == "linear_regression":
            self.calculate_error = self.linear_regression      
        elif plr == "linear_interpolation":
            self.calculate_error = self.linear_interpolation
        else:
            print("wrong plr")
    
    def segment_plot(self,dim = 0):
        colors = {0:"r",1:"b"}
        k = 0
        for i in range(len(self.segments)):
            plot_data = self.segments[i].data[:,0]
            plt.plot(np.arange(k,k+len(plot_data)),plot_data, colors[i%2])
            k = len(plot_data) + k
        plt.show
    
    def create_segment(self,data):
        return segment(data,self.calculate_error(data))

class plr:
    """
    """
    def linear_regression(self,data):
        A = np.vstack([np.arange(len(data)),np.ones(len(data))]).T
        residuals = np.linalg.lstsq(A,data,rcond=None)[1]
        residuals = 0 if len(residuals) == 0 else residuals.mean()
        return residuals
    
    def linear_interpolation(self,data):
        steps = np.arange(0,len(data),1)
        pred = np.interp(steps,data[0,:],data[len(data)-1,:])
        sqrd_error = (data - pred)**2
        return np.mean(sqrd_error)
        

class top_down(estimator,plr):
    """
    """
    def __init__(self):
        estimator.__init__(self)
        self.algorithm = "top down"

    def improvement_in_splitting(self,data,i):
        return(self.calculate_error(data[:i]) + self.calculate_error(data[i:]))

    def top_down_split(self,data,max_error):
        best_so_far = np.inf
        for i in range(2,len(data)-1):
            improvement_in_approximation = self.improvement_in_splitting(data,i)
            if improvement_in_approximation < best_so_far:
                best_so_far = improvement_in_approximation
                break_point = i
        
        if self.calculate_error(data[:break_point]) > max_error and len(data[:break_point]) > 2:
            self.top_down_split(data[:break_point],max_error)             
        else: 
            self.error += self.calculate_error(data[:break_point])
            self.segments.append(self.create_segment(data[:break_point]))
            
        if self.calculate_error(data[break_point:]) > max_error and len(data[break_point:]) > 2:
            self.top_down_split(data[break_point:],max_error) 
        else: 
            self.error += self.calculate_error(data[break_point:])
            self.segments.append(self.create_segment(data[break_point:]))
    
    def fit(self,data,max_error,plr = "linear_interpolation"):
        estimator.fit(self,data,max_error,plr)  
        self.segment_borders.append(0)
        self.top_down_split(data,max_error)               
        for i in range(len(self.segments)):
            temp_data = self.segments[i].data
            self.segment_borders.append(len(temp_data)+self.segment_borders[-1])
            self.labels[self.segment_borders[-2]:self.segment_borders[-1]] = i            
        del self.segment_borders[-1]
            
class bottom_up(estimator,plr):
    
    def __init__(self):
        estimator.__init__(self)
        self.algorithm  = "bottom up"
    
    def fit(self,data,max_error,plr = "linear_interpolation"):
        estimator.fit(self,data,max_error,plr)
                       
        for i in range(0,len(data)-2,2):
            self.segments.append(self.create_segment(data[i:i+2]))
        
        merge_cost = np.zeros(len(self.segments)-1)        
        for i in range(len(self.segments)-1):
            merge_cost[i] = self.calculate_error(np.concatenate((self.segments[i].data,self.segments[i+1].data)))
        
        while True:
            try: 
                if np.min(merge_cost) < max_error:
                    index = np.argmin(merge_cost)
                    self.segments[index].data = np.concatenate((self.segments[index].data,self.segments[index+1].data))
                    del self.segments[index+1]
                    merge_cost = np.delete(merge_cost,index)
                    if index < (len(self.segments)-1):
                        merge_cost[index] = self.calculate_error(np.concatenate((self.segments[index].data,self.segments[index+1].data)))
                    if index != 0:
                        merge_cost[index-1] = self.calculate_error(np.concatenate((self.segments[index-1].data,self.segments[index].data)))                    
                else:
                    break
            except:
                break
        self.labels = np.concatenate([np.ones(len(self.segments[i].data))*i for i in range(len(self.segments))])
        borders = np.nonzero(self.labels != np.roll(self.labels,1))
        self.segment_borders.extend(borders[0].tolist())
            
class sliding_window(estimator,plr):
    
    def __init__(self):
        estimator.__init__(self)
        self.algorithm  = "sliding window"
    
    def fit(self,data,max_error,plr = "linear_interpolation"):
        estimator.fit(self,data,max_error,plr)                      
        anchor = 0      
        finished = False
        k = 0
        while not finished:    
            i = 2
            while self.calculate_error(data[anchor:anchor + i]) < max_error and anchor + i <= len(data):
                i += 1
            self.segments.append(self.create_segment(data[anchor:anchor + i -1]))
            self.labels[anchor:(anchor + (i - 1))] = k
            self.segment_borders.append(anchor + (i - 1))
            self.error += self.calculate_error(data[anchor:anchor + i - 1])
            k += 1
            anchor = anchor + (i - 1)
            if anchor >= len(data):
                finished = True        
                
# class SWAB(estimator,plr):
    
#     def __init__(self):
#         estimator.__init__(self)
#         self.algorithm = "SWAB"
        
#     def fit(self,data,max_error,plr = "linear_interpolation",buffer_size = 100):
#         estimator.fit(self,data,max_error,plr)
#         i = 0
#         while  
            
