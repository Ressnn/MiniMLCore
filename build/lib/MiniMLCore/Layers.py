# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:02:15 2020

@author: Pranav Devarinti
"""
from abc import ABC, abstractmethod
from OperationsManagers.CPU import CPU
from sympy import Symbol, Derivative
import numpy as np
import math

class Layer(ABC):
    def __init__(self,operations=CPU):
        self.output_shape = self.get_output_shape()
        self.operations = operations
    
    @abstractmethod
    def build(self,input_shape: tuple):
        self.input_shape = input_shape
        pass
    
    @abstractmethod
    def compute(self,inputs):
        pass
    @abstractmethod
    def get_output_shape(self):
        pass
    @abstractmethod
    def trainable_weight_derivatives(self):
        pass
    @abstractmethod
    def apply_changes(self):
        pass

class Dense(Layer):
    def __init__(self,units):
        self.units = units
    def build(self, input_shape :list,operations=CPU):
        self.operations = operations
        self.input_shape = input_shape
        bound = np.sqrt(6)/np.sqrt((np.product(self.input_shape)+self.units))
        self.weights = self.operations.default_array(np.random.uniform(size=(self.units,int(input_shape[0])),low=-bound,high=bound))
        self.baises = self.operations.default_array(np.random.uniform(-bound,bound))
        super().__init__(self.operations)
    def compute(self,inputs):
        return self.operations.add(self.operations.dot(self.weights, inputs),self.baises)
    def get_output_shape(self):
        return list(self.compute(self.operations.get_random_method()(size=self.input_shape)).shape)
    def trainable_weight_derivatives(self,inputs,deltas):
        trainable_weights = dict()
        trainable_weights["Previous_Layer"] = self.operations.dot(self.weights.transpose(),deltas)
        trainable_weights["Weights"] = self.operations.reshape(inputs,(1,-1))*self.operations.reshape(deltas,(-1,1))
        trainable_weights["Baises"] = deltas
        return trainable_weights
    def apply_changes(self,trainable_weights_dict):
        self.weights = trainable_weights_dict["Weights"]+self.weights
        self.baises = trainable_weights_dict["Baises"]+self.baises
        
        
class Simple_Activation(Layer):
    def __init__(self,input_shape,operations):
        self.input_shape = input_shape
        self.operations = operations
    def get_output_shape(self):
        return self.input_shape
    def trainable_weight_derivatives(self,inputs,deltas):
        trainable_weights = dict()
        trainable_weights["Previous_Layer"] = self.derivative(inputs)*deltas
        return trainable_weights
    def apply_changes(self,changes):
        pass
    @abstractmethod
    def derivative(inputs):
        pass

    # Remember that we still also need to define compute
    
class ReLU(Simple_Activation):
    def __init__(self):
        pass
    def build(self,input_shape: tuple,operations=CPU):
        super().__init__(input_shape,operations)
    def derivative(self,x):
        return self.operations.where(x <=0,0,1)
    def compute(self, x):
        return self.operations.where(x <=0,0,x)

class SeLU(Simple_Activation):
    
    def __init__(self):
        pass
    def build(self,input_shape: tuple,operations=CPU):
        self.alpha_value = 1.6732632423543772848170429916717
        self.lambda_value = 1.0507009873554804934193349852946
        super().__init__(input_shape,operations)
    def derivative(self,x):
        return self.operations.where(x <=0,self.alpha_value*(math.e**x),1)
    def compute(self, x):
        return self.operations.where(x <=0,self.lambda_value*(self.alpha_value*(math.e**x))-self.alpha_value,self.lambda_value*x)

class Sigmoid(Simple_Activation):
    def __init__(self):
        pass
    def build(self,input_shape: tuple,operations=CPU):
        super().__init__(input_shape,operations)
    def derivative(self,x):
        return self.compute(x)*(1-self.compute(x))
    def compute(self, x):
        return 1/(1 + math.e**(-x))


# In[]
"""
import cupy
from OperationsManagers.CUDA import CUDAAcceleratedOperations as CUDA
#D = Dense([3],5,operations=CUDA)
#print(D.trainable_weight_derivatives(cupy.array([1,2,3]),cupy.array([1,2,3,2,1])))

R = SeLU()
R.build([3],operations=CUDA)
V = R.trainable_weight_derivatives(cupy.array([1,2,3]),cupy.array([1,2,1]))
print(V)
"""