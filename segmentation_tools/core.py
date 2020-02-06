# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 11:41:27 2020

@author: OediP
"""

import pandas as pd
import numpy as np
#import pdb
#to do:
#swab implementation
#bottom up , wenn keine segemente
#absolute error
#rmse error
#linear interpolation
#indices instead of data
#max error, error metrics, total_max error, error_per_segment
#num segments, labels,
#interplolate, regress
# errors, absolute difference

def add(a,b):
    return a+b

class segment:
    def __init__(self,data):
        self.data = data
    
        
class estimator:
    """
    """
    def __init__(self):
        #self.data = None
        self.segments = None
        self.max_error = None
        self.labels = None
        self.algorithm = None
        self.error = 0
        self.plr = None
        self.calculate_error = None
        self.segment_borders = list()
        self.labels = None
        
    def fit(self,data,max_error,plr = "linear_regression"):
        self.labels = np.zeros(data.shape[0])
        #self.data = data
        self.max_error = max_error
        self.plr = plr
        self.row_num = len(data)
        self.segments = list()
        if plr == "linear_regression":
            self.calculate_error = self.linear_regression      
        elif plr == "linear_interpolation":
            self.calculate_error = self.linear_interpolation
        else:
            print("wrong plr")
            
    def calculate_error_r(self,data):
        A = np.vstack([np.arange(len(data)),np.ones(len(data))]).T
        residuals = np.linalg.lstsq(A,data,rcond=None)[1]
        residuals = 0 if len(residuals) == 0 else residuals.mean() 
        return residuals
    
    def create_segment(self,data):
        return segment(data)

class plr:
    def linear_regression(self,data):
        #pdb.set_trace()
        A = np.vstack([np.arange(len(data)),np.ones(len(data))]).T
        residuals = np.linalg.lstsq(A,data,rcond=None)[1]
        residuals = 0 if len(residuals) == 0 else residuals.mean()
        return residuals
    
    def linear_interpolation(self):
        pass
    

class top_down(estimator,plr):
    def __init__(self):
        estimator.__init__(self)
        self.algorithm = "top down"

    def improvement_in_splitting(self,data,i):
        return(self.calculate_error(data[:i]) + self.calculate_error(data[i:]))

    def top_down_split(self,data,max_error):
        best_so_far = np.inf
        for i in range(2,self.row_num - 1):
            improvement_in_approximation = self.improvement_in_splitting(data,i)
            if improvement_in_approximation < best_so_far:
                best_so_far = improvement_in_approximation
                break_point = i
        first = data[:break_point]
        second = data[break_point:] 
        if self.calculate_error(data[:break_point]) > max_error:
            first_sub,second_sub = self.top_down_split(data[:break_point],max_error) 
#            self.segments.append(self.create_segment(first_sub))
#            self.segments.append(self.create_segment(second_sub))            
            self.labels[:break_point] = 1
            self.labels[break_point:] = 2
            self.error += self.calculate_error(first_sub)
            self.error += self.calculate_error(second_sub)
        if self.calculate_error(data[break_point:]) > max_error:
            first_sub,second_sub = self.top_down_split(data[break_point:],max_error) 
#            self.segments.append(self.create_segment(first_sub))
#            self.segments.append(self.create_segment(second_sub)) 
            self.error += self.calculate_error(first_sub)
            self.error += self.calculate_error(second_sub)        
        return (first,second)
    
    def fit(self,data,max_error,plr = "linear_regression"):
        estimator.fit(self,data,max_error,plr)        
        self.top_down_split(data,max_error)

class bottom_up(estimator,plr):
    
    def __init__(self):
        estimator.__init__(self)
        self.algorithm  = "bottom up"
    
    def fit(self,data,max_error,plr = "linear_regression"):
        estimator.fit(self,data,max_error,plr)
                       
        for i in range(0,len(data)-2,2):
            self.segments.append(self.create_segment(data[i:i+2]))
        merge_cost = np.zeros(len(self.segments)-1)
        
        for i in range(len(self.segments)-1):
            merge_cost[i] = self.calculate_error(np.concatenate((self.segments[i].data,self.segments[i+1].data)))
        
        while np.min(merge_cost) < max_error:
            index = np.argmin(merge_cost)
            self.segments[index].data = np.concatenate((self.segments[index].data,self.segments[index+1].data))
            del self.segments[index+1]
            merge_cost = np.delete(merge_cost,index)
            if index < (len(self.segments)-1):
                merge_cost[index] = self.calculate_error(np.concatenate((self.segments[index].data,self.segments[index+1].data)))
            if index != 0:
                merge_cost[index-1] = self.calculate_error(np.concatenate((self.segments[index-1].data,self.segments[index].data)))
            
class sliding_window(estimator,plr):
    
    def __init__(self):
        estimator.__init__(self)
        self.algorithm  = "sliding window"
    
    def fit(self,data,max_error,plr = "linear_regression"):
        estimator.fit(self,data,max_error,plr)                      
        anchor = 0      
        finished = False
        k = 0
        while not finished:    
            i = 2
            while self.calculate_error(data[anchor:anchor + i]) < max_error and anchor + i <= len(data):
                i += 1
            self.labels[anchor:(anchor + (i - 1))] = k
            self.segment_borders.append(anchor + (i - 1))
            self.error += self.calculate_error(data[anchor:anchor + i - 1])
            k += 1
            anchor = anchor + (i - 1)
            if anchor >= len(data):
                finished = True        
