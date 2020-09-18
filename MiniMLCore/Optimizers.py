from abc import ABC,abstractmethod
from Losses import MeanSquaredError
from OperationsManagers.CPU import CPU
from Model import Sequential
from Layers import *
import numpy as np
from itertools import islice 
from numba import jit

class Derivatives():
    def __init__(self,derivatives):
        self.derivatives = derivatives    
    def _add(self,other):
        return_list = []
        for i in self.derivatives:
            return_list.append(dict([(k, i[k]+other) for k in set(i)]))
        return Derivatives(return_list)
    def __add__(self,other):
        if (type(other) == float or type(other) == int):
            return self._add(other)
        return_list = []
        for i in zip(self.derivatives,other.derivatives):
            return_list.append(dict([(k, i[0][k]+i[1][k]) for k in set(i[0]) & set(i[1])]))
        return Derivatives(return_list)
    def _sub(self,other):
        return_list = []
        for i in self.derivatives:
            return_list.append(dict([(k, i[k]-other) for k in set(i)]))
        return Derivatives(return_list)
    def __sub__(self,other):
        if (type(other) == float or type(other) == int):
            return self._sub(other)
        return_list = []
        for i in zip(self.derivatives,other.derivatives):
            return_list.append(dict([(k, i[0][k]-i[1][k]) for k in set(i[0]) & set(i[1])]))
        return Derivatives(return_list)
    def _mul(self,other):
        return_list = []
        for i in self.derivatives:
            return_list.append(dict([(k, i[k]*other) for k in set(i)]))
        return Derivatives(return_list)
    def __mul__(self,other):
        if (type(other) == float or type(other) == int):
            return self._mul(other)
        return_list = []
        for i in zip(self.derivatives,other.derivatives):
            return_list.append(dict([(k, i[0][k]*i[1][k]) for k in set(i[0]) & set(i[1])]))
        return Derivatives(return_list)
    def __div__(self,other):
        return_list = []
        for i in self.derivatives:
            return_list.append(dict([(k, i[k]/other) for k in set(i)]))
        return Derivatives(return_list)
    def __truediv__(self,other):
        if (type(other) == float or type(other) == int):
            return self.__div__(other)
        return_list = []
        for i in zip(self.derivatives,other.derivatives):
            return_list.append(dict([(k, i[0][k]/i[1][k]) for k in set(i[0]) & set(i[1])]))
        return Derivatives(return_list)
    def zeroes(self):
        return self.__sub__(self)
    def sqrt(self):
        return_list = []
        for i in self.derivatives:
            return_list.append(dict([(k, i[k]**(1/2)) for k in set(i)]))
        return Derivatives(return_list)
    
class GradientCalculator():
    def __init__(self,model :Sequential,loss=MeanSquaredError):
        self.model = model
        self.operations = model.operations
        self.loss = loss
    def calculate_gradients(self,inputs,expected_outputs):
        layerByLayer = self.model.layerByLayerPredict(inputs)
        deltas = self.loss.computeGradients(expected_outputs,layerByLayer[-1])
        derivatives_list = []
        for layer_index in np.arange(len(self.model.Network)-1,-1,-1):
            layer = self.model.Network[layer_index]
            inputs = layerByLayer[layer_index]
            derivatives = layer.trainable_weight_derivatives(inputs,deltas)
            derivatives_list.append(derivatives)
            deltas = derivatives["Previous_Layer"]
        return Derivatives(derivatives_list[::-1])
    def calculate_minibatches(self,inputs,outputs):
        grads = self.calculate_gradients(inputs[0],outputs[0])
        for i in zip(inputs[1::],outputs[1::]):
            grads += self.calculate_gradients(i[0],i[1])
        return grads/len(inputs)

class Optimizer(ABC,GradientCalculator):
    def __init__(self,model :Sequential,loss=MeanSquaredError,batch_size=32):
        super().__init__(model,loss=loss)
        self.batch_size = batch_size
        self.loss = loss
    def batch_maker(self,inputs,outputs):
        length = len(inputs)
        split_sizes = []
        for i in range(0,int(np.floor(length/self.batch_size))):
            split_sizes.append(self.batch_size)
        split_sizes.append(length%self.batch_size)
        return [list(islice(inputs, elem)) for elem in split_sizes],[list(islice(outputs, elem)) for elem in split_sizes]

        
        
class Adam(Optimizer):
    def __init__(self,model :Sequential,loss=MeanSquaredError,batch_size=32,beta1=.9,beta2=.999,epsilon=1e-10,learning_rate=.001):
        super().__init__(model,loss=loss,batch_size=batch_size)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.m_t = None
        self.v_t = None
        self.FirstRun = True
        
    def first_run(self,inputs,outputs):
        sample = self.calculate_gradients(inputs[0],outputs[0])
        self.m_t = sample.zeroes()
        self.v_t = sample.zeroes()
    
    def train_step(self,inputs,outputs):
        if self.FirstRun:
            self.FirstRun = False
            self.first_run(inputs,outputs)
        batchedx,batchedy = self.batch_maker(inputs, outputs)
        for i in zip(batchedx,batchedy):
            g_t = self.calculate_minibatches(i[0],i[1])
            self.m_t = self.m_t*self.beta1+g_t*(1-self.beta1)
            self.v_t = self.v_t*self.beta2+g_t*g_t*(1-self.beta2)
            m_t_cap = self.m_t/(1-self.beta1)
            v_t_cap = self.v_t/(1-self.beta2)
            self.adjs = (m_t_cap/(v_t_cap.sqrt()+self.epsilon))*(self.learning_rate)
            self.model.applyChanges(self.adjs.derivatives)
    def train(self,inputs,outputs,epochs=1,calculate_loss=True):
        loss = []
        inputs = self.operations.default_array(inputs)
        outputs = self.operations.default_array(outputs)
        for epoch in range(0,epochs):
            print("Epoch:"+str(epoch))
            self.train_step(inputs,outputs)
            if calculate_loss:
                loss.append(self.operations.average(self.loss.computeLoss(self.model.batchPredict(inputs),outputs.reshape(-1,1)),layer=None))
                print(loss[-1])
        return loss
# In[]
from OperationsManagers.CUDA import CUDAAcceleratedOperations as CUDA
import cupy
Model = Sequential([1])
Model.add(Dense(100))
Model.add(Sigmoid())
Model.add(Dense(400))
Model.add(Sigmoid())
Model.add(Dense(1))
Model.build()
# In[]
O = Adam(Model,learning_rate=.001,batch_size=32000)
inputs = np.arange(0,np.pi,.01)
outputs = np.cos(inputs)
A = O.train(inputs,outputs,5000,True)

# In[]
import matplotlib.pyplot as plt
inputs = np.arange(0,np.pi,.01)
outputs = np.cos(inputs)
plt.plot(inputs,outputs)
plt.plot(inputs,Model.batchPredict(inputs).reshape(-1))

# In[]
"""
B = G.calculate_gradients(a,a).derivatives
A = G.calculate_gradients(a,a)/1
V = A.derivatives


import keras
Model2 = keras.Sequential()
Model2.add(keras.layers.Dense(400,activation="sigmoid"))
Model2.add(keras.layers.Dense(100,activation="sigmoid"))
Model2.add(keras.layers.Dense(1))
Model2.compile(optimizer="adam",loss="mse")
Model2.fit(inputs.reshape(-1,1),outputs.reshape(-1,1),epochs=1000)
"""